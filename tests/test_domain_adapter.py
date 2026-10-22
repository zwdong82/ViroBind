"""CPU regression tests with anonymous random tensors, never research data."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


RUNTIME_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "numpy", "pandas", "sklearn")
)
if RUNTIME_AVAILABLE:
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
    from virobind import base
    from virobind import model as v5


@unittest.skipUnless(RUNTIME_AVAILABLE, "CPU model dependencies are not installed")
class JointDomainAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.features = torch.randn(4, 12)
        self.proteins = torch.randn(4, 10)
        self.domains = torch.tensor([0, 1, 0, 1])
        self.tokens = torch.randn(2, 5, 10)
        self.mask = torch.tensor([[True] * 5, [True, True, True, False, False]])
        self.pair_index = torch.tensor([0, 1, 0, 1])

    def make_model(self, *, mode="full_bilinear", adapter=True, residue=True):
        args = SimpleNamespace(
            drug_fusion_mode="residual_multiview",
            drug_ecfp_dim=5, drug_maccs_dim=3, drug_rdkit_dim=4,
            drug_view_mask="1,1,1",
            protein_fusion_mode="esmc_only", protein_esmc_dim=10,
            token_root="synthetic-unused-path" if residue else "",
            residue_rank_only=False, pair_feature_mode=mode,
            rank_train_scope="rank_head",
        )
        v5.configure_from_namespace(args)
        return v5.ViroBindV5Decoupled(
            "ecfp_esmc", 12, 1, 10, hidden_dim=8, bilinear_dim=3,
            dropout=0.0, use_domain_adapter=adapter,
        )

    def forward(self, model, domains=None):
        return model(
            self.features, None, self.proteins,
            self.domains if domains is None else domains,
            self.tokens if model.use_residue_tokens else None,
            self.mask if model.use_residue_tokens else None,
            self.pair_index if model.use_residue_tokens else None,
        )

    def test_both_heads_share_selected_adapted_normalized_interaction(self):
        model = self.make_model().eval()
        captured = {}
        handles = []

        def capture(name):
            def hook(module, inputs, output):
                captured[name] = (inputs[0], output)
            return hook

        for name in ("pair_norm", "human_adapter", "virus_adapter", "cls_head", "rank_head"):
            handles.append(getattr(model, name).register_forward_hook(capture(name)))
        try:
            self.forward(model)
        finally:
            for handle in handles:
                handle.remove()
        pair = captured["pair_norm"][1]
        torch.testing.assert_close(captured["human_adapter"][0], pair)
        torch.testing.assert_close(captured["virus_adapter"][0], pair)
        selected = torch.where(
            self.domains.bool().view(-1, 1),
            captured["virus_adapter"][1], captured["human_adapter"][1],
        )
        torch.testing.assert_close(captured["cls_head"][0], selected)
        self.assertIs(captured["cls_head"][0], captured["rank_head"][0])
        self.assertEqual(pair.shape, (4, 35))

    def test_all_interaction_ablation_shapes_and_residue_modes_run(self):
        dimensions = {
            "concat_only": 16, "concat_product": 24,
            "concat_product_absdiff": 32, "full_bilinear": 35,
        }
        for mode, dimension in dimensions.items():
            for residue in (False, True):
                with self.subTest(mode=mode, residue=residue):
                    model = self.make_model(mode=mode, residue=residue)
                    output = self.forward(model)
                    self.assertEqual(model.virus_adapter.net[0].weight.numel(), dimension)
                    for head in ("cls", "rank"):
                        self.assertEqual(output[head].shape, (4,))
                        self.assertTrue(torch.isfinite(output[head]).all())
                    (output["cls"].sum() + output["rank"].sum()).backward()
                    self.assertGreater(
                        sum(p.grad.abs().sum().item() for p in model.drug_encoder.parameters() if p.grad is not None),
                        0,
                    )

    def test_only_selected_domain_adapter_receives_gradient(self):
        for domain, active, inactive in (
            (0, "human_adapter", "virus_adapter"),
            (1, "virus_adapter", "human_adapter"),
        ):
            with self.subTest(domain=domain):
                model = self.make_model()
                output = self.forward(model, torch.full((4,), domain))
                (output["cls"].sum() + output["rank"].sum()).backward()
                active_grad = sum(p.grad.abs().sum().item() for p in getattr(model, active).parameters())
                inactive_grad = sum(p.grad.abs().sum().item() for p in getattr(model, inactive).parameters())
                self.assertGreater(active_grad, 0)
                self.assertEqual(inactive_grad, 0)

    def test_noadapt_has_no_domain_dependent_output(self):
        model = self.make_model(adapter=False).eval()
        human = self.forward(model, torch.zeros(4, dtype=torch.long))
        virus = self.forward(model, torch.ones(4, dtype=torch.long))
        self.assertFalse(hasattr(model, "human_adapter"))
        for head in ("cls", "rank"):
            torch.testing.assert_close(human[head], virus[head])

    def test_rank_head_only_update_preserves_classification_and_upstream(self):
        model = self.make_model().eval()
        v5.set_rank_trainable(model, "rank_head")
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        cls_before = self.forward(model)["cls"].detach().clone()
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
        optimizer.zero_grad()
        self.forward(model)["rank"].sum().backward()
        optimizer.step()
        changed = {
            name for name, tensor in model.state_dict().items()
            if not torch.equal(tensor, before[name])
        }
        self.assertTrue(changed)
        self.assertTrue(all(name.startswith("rank_head.") for name in changed))
        torch.testing.assert_close(self.forward(model)["cls"], cls_before, rtol=0, atol=0)

    def test_ranking_ablation_updates_existing_virus_adapter(self):
        for scope in ("rank_head_adapter", "rank_head_pair"):
            with self.subTest(scope=scope):
                model = self.make_model()
                v5.set_rank_trainable(model, scope)
                trainable = {name for name, p in model.named_parameters() if p.requires_grad}
                expected_prefixes = ["rank_head.", "virus_adapter."]
                if scope == "rank_head_pair":
                    expected_prefixes += ["pair_norm.", "drug_bilinear.", "prot_bilinear."]
                expected = {
                    name for name, _ in model.named_parameters()
                    if any(name.startswith(prefix) for prefix in expected_prefixes)
                }
                self.assertEqual(trainable, expected)
                self.assertFalse(hasattr(model, "rank_adapter"))
                before = model.virus_adapter.net[-1].weight.detach().clone()
                optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
                optimizer.zero_grad()
                self.forward(model, torch.ones(4, dtype=torch.long))["rank"].sum().backward()
                optimizer.step()
                self.assertFalse(torch.equal(before, model.virus_adapter.net[-1].weight))

    def test_ranking_training_enables_only_requested_module_modes(self):
        model = self.make_model()
        args = SimpleNamespace(
            rank_train_scope="rank_head_adapter", feature_preset="ecfp_esmc",
            grad_clip=5.0,
        )
        v5.set_rank_trainable(model, args.rank_train_scope)
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
        batch = {"y": torch.tensor([1., 0., 1., 0.]), "group": torch.zeros(4)}
        with patch.object(v5, "batch_forward", side_effect=lambda *a: self.forward(model)):
            with patch.object(v5, "rank_loss", side_effect=lambda logits, *a: logits.square().mean()):
                v5.train_one_epoch(
                    model, [batch], {}, torch.device("cpu"), args, optimizer,
                    torch.tensor(1.0), "rank",
                )
        self.assertTrue(model.virus_adapter.training)
        self.assertTrue(model.rank_head.training)
        self.assertFalse(model.human_adapter.training)
        self.assertFalse(model.drug_encoder.training)
        self.assertFalse(model.cls_head.training)

    def test_new_checkpoint_roundtrip_and_legacy_adapter_rejection(self):
        model = self.make_model().eval()
        restored = self.make_model().eval()
        restored.load_state_dict(model.state_dict())
        for head in ("cls", "rank"):
            torch.testing.assert_close(self.forward(model)[head], self.forward(restored)[head])
        state = model.state_dict()
        for domain in ("human_adapter", "virus_adapter"):
            legacy = base.DomainAdapter(8, 32, 0.0)
            for name, tensor in legacy.state_dict().items():
                state[f"{domain}.{name}"] = tensor
        for strict in (True, False):
            with self.subTest(strict=strict):
                with self.assertRaisesRegex(RuntimeError, "legacy protein-domain adapter"):
                    restored.load_state_dict(state, strict=strict)

    def test_invalid_ranking_scope_is_rejected_before_freezing(self):
        model = self.make_model(adapter=False)
        with self.assertRaisesRegex(ValueError, "requires use_domain_adapter=1"):
            v5.set_rank_trainable(model, "rank_head_adapter")
        self.assertTrue(all(p.requires_grad for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
