const githubUrl = "https://github.com/zwdong82/ViroBind";
const modelCardUrl = `${githubUrl}/blob/main/Pretrained_models/ViroBind/MODEL_CARD.md`;

const Arrow = () => <span aria-hidden="true">↗</span>;

export default function Home() {
  return (
    <main>
      <nav className="nav shell" aria-label="Primary navigation">
        <a className="brand" href="#top" aria-label="ViroBind home">
          <span className="brand-mark" aria-hidden="true">
            <i />
            <i />
            <i />
          </span>
          <span>ViroBind</span>
        </a>
        <div className="nav-links">
          <a href="#capabilities">Capabilities</a>
          <a href="#workflow">Workflow</a>
          <a href="#responsible-use">Responsible use</a>
        </div>
        <a className="nav-cta" href={githubUrl} target="_blank" rel="noreferrer">
          View source <Arrow />
        </a>
      </nav>

      <section className="hero shell" id="top">
        <div className="hero-copy">
          <div className="eyebrow"><span /> Research model · Noncommercial use</div>
          <h1>Turn compound libraries into <em>focused antiviral hypotheses.</em></h1>
          <p className="hero-lede">
            ViroBind is a deep-learning framework for drug–virus interaction prediction and
            candidate ranking, built to move from compatible molecular and protein features
            to a sharper experimental shortlist.
          </p>
          <div className="hero-actions">
            <a className="button primary" href={githubUrl} target="_blank" rel="noreferrer">
              Explore on GitHub <Arrow />
            </a>
            <a className="button secondary" href={modelCardUrl} target="_blank" rel="noreferrer">
              Read the model card
            </a>
          </div>
          <div className="hero-note">
            <span className="pulse" /> Built for hypothesis generation—not clinical decisions.
          </div>
        </div>

        <div className="instrument" aria-label="Illustrative ViroBind prediction interface">
          <div className="instrument-top">
            <span>Prediction surface</span>
            <span className="status"><i /> MODEL READY</span>
          </div>
          <div className="pair-row">
            <div className="entity drug-entity">
              <div className="entity-kicker">COMPOUND</div>
              <strong>VB–D042</strong>
              <div className="fingerprint" aria-hidden="true">
                {Array.from({ length: 36 }).map((_, index) => <i key={index} />)}
              </div>
              <small>ECFP4 · MACCS · RDKit2D</small>
            </div>
            <div className="link-node" aria-hidden="true"><span>×</span></div>
            <div className="entity protein-entity">
              <div className="entity-kicker">VIRAL PROTEIN</div>
              <strong>VB–P017</strong>
              <div className="protein-trace" aria-hidden="true">
                <i /><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /><i />
              </div>
              <small>ESMC global · residue tokens</small>
            </div>
          </div>
          <div className="score-panel">
            <div>
              <span>Illustrative interaction score</span>
              <strong>0.873</strong>
            </div>
            <div className="score-meter" aria-hidden="true"><i /></div>
            <p>High-priority research signal</p>
          </div>
          <div className="instrument-foot">
            <span>Classification head</span>
            <span>Ranking head</span>
            <span>Chunked screening</span>
          </div>
        </div>
      </section>

      <section className="metrics shell" aria-label="Model representation summary">
        <div><strong>Multi-view</strong><span>drug representation</span></div>
        <div><strong>Multi-scale</strong><span>protein representation</span></div>
        <div><strong>2</strong><span>task-specific checkpoints</span></div>
        <div><strong>GPU / CPU</strong><span>compatible inference</span></div>
      </section>

      <section className="section shell" id="capabilities">
        <div className="section-heading">
          <div>
            <span className="section-index">01 / CAPABILITIES</span>
            <h2>Two tasks. One consistent representation.</h2>
          </div>
          <p>Use ViroBind for targeted pair assessment or memory-aware screening across a larger compound library.</p>
        </div>
        <div className="capability-grid">
          <article className="capability-card lime">
            <div className="card-number">01</div>
            <div className="card-visual pair-visual" aria-hidden="true">
              <span>D</span><i /><span>P</span>
            </div>
            <h3>CPI pair prediction</h3>
            <p>Estimate a binary interaction probability for prepared compound–protein pairs using the classification checkpoint.</p>
            <ul>
              <li>Paired anonymous IDs</li>
              <li>Compatible precomputed feature banks</li>
              <li>Batch-ready CSV output</li>
            </ul>
            <code>virobind-predict</code>
          </article>
          <article className="capability-card blue">
            <div className="card-number">02</div>
            <div className="card-visual rank-visual" aria-hidden="true">
              <i /><i /><i /><i /><i />
            </div>
            <h3>Large-library ranking</h3>
            <p>Score compounds against one or more protein targets and retain the highest-priority candidates for review.</p>
            <ul>
              <li>Memory-mapped drug features</li>
              <li>Chunked scoring at scale</li>
              <li>Target-specific top-k results</li>
            </ul>
            <code>virobind-screen</code>
          </article>
        </div>
      </section>

      <section className="section workflow-section" id="workflow">
        <div className="shell">
          <div className="section-heading compact">
            <div>
              <span className="section-index">02 / WORKFLOW</span>
              <h2>From raw candidates to a ranked research queue.</h2>
            </div>
          </div>
          <div className="workflow-grid">
            <article>
              <span className="step">STEP 01</span>
              <div className="workflow-icon">01</div>
              <h3>Prepare representations</h3>
              <p>Generate ECFP4, MACCS and standardized RDKit2D descriptors for compounds, plus ESMC global and residue-level features for proteins.</p>
            </article>
            <article>
              <span className="step">STEP 02</span>
              <div className="workflow-icon">02</div>
              <h3>Run the right head</h3>
              <p>Choose classification scores for specific pairs or ranking scores to order a broader candidate library.</p>
            </article>
            <article>
              <span className="step">STEP 03</span>
              <div className="workflow-icon">03</div>
              <h3>Review and validate</h3>
              <p>Use the output as a prioritization signal, then apply domain review and independent experimental validation before drawing conclusions.</p>
            </article>
          </div>
        </div>
      </section>

      <section className="section shell architecture-section">
        <div className="section-heading">
          <div>
            <span className="section-index">03 / MODEL LOGIC</span>
            <h2>Multimodal evidence, fused for prioritization.</h2>
          </div>
          <p>The released checkpoints encode the architecture needed for inference and use the human-pretraining / virus-fine-tuning protocol.</p>
        </div>
        <div className="architecture">
          <div className="architecture-source">
            <span>DRUG BRANCH</span>
            <strong>ECFP4 + MACCS + RDKit2D</strong>
            <small>combined molecular representation</small>
          </div>
          <div className="architecture-connector"><i /><span>feature fusion</span><i /></div>
          <div className="architecture-source">
            <span>PROTEIN BRANCH</span>
            <strong>ESMC global + residue context</strong>
            <small>global and residue-level representation</small>
          </div>
          <div className="architecture-arrow" aria-hidden="true">→</div>
          <div className="architecture-output">
            <span>TASK HEADS</span>
            <strong>Classify</strong>
            <strong>Rank</strong>
          </div>
        </div>
      </section>

      <section className="responsible" id="responsible-use">
        <div className="shell responsible-grid">
          <div>
            <span className="section-index light">04 / RESPONSIBLE USE</span>
            <h2>A signal for the lab—not a substitute for it.</h2>
          </div>
          <div className="responsible-copy">
            <p>ViroBind is intended for noncommercial academic and public-interest research. Scores help generate hypotheses; they do not establish physical binding, antiviral activity, safety or efficacy.</p>
            <div className="limits">
              <span>No clinical decisions</span>
              <span>No patient-level use</span>
              <span>Independent validation required</span>
            </div>
            <a href={modelCardUrl} target="_blank" rel="noreferrer">Review limitations and intended use <Arrow /></a>
          </div>
        </div>
      </section>

      <section className="cta-section shell">
        <div>
          <span className="section-index">READY TO EXPLORE?</span>
          <h2>Inspect the model. Reproduce the pipeline. Test your hypotheses.</h2>
        </div>
        <a className="button primary dark" href={githubUrl} target="_blank" rel="noreferrer">
          Open ViroBind on GitHub <Arrow />
        </a>
      </section>

      <footer className="footer shell">
        <a className="brand" href="#top"><span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>ViroBind</a>
        <p>Deep-learning drug–virus interaction prediction and candidate prioritization.</p>
        <span>PolyForm Noncommercial 1.0.0</span>
      </footer>
    </main>
  );
}
