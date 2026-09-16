"""How a completion is scored: reward terms parsed from config (:mod:`.spec`), the samples a scorer
reads (:mod:`.samples`), the scorers behind the external sources — a generative judge
(:mod:`.judge`), a served reward model (:mod:`.reward_model`) — their composition into one reward
(:mod:`.composer`), the TRL reward-function adapters (:mod:`.functions`), the RLVR graders
(:mod:`.verifiable`) and rule-based answer matching (:mod:`.matching`). Import the submodule you
need: :mod:`.spec` stays light enough for the argument dataclasses, the scorers do not."""
