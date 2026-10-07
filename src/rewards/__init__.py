"""How a completion is scored: reward terms parsed from config (:mod:`.terms`), the sample a scorer
reads and the views of it a judge sees (:mod:`.samples`), the scorers behind the external sources
(:mod:`.scorers`: the contract, a generative judge, a served reward model and their catalog), their
composition and settlement into one reward (:mod:`.composer`), the TRL reward-function adapters
(:mod:`.functions`) and the rule-based graders (:mod:`.graders`: answer matching, the RLVR graders).
Import the submodule you need: :mod:`.terms` stays light enough for the argument dataclasses, the
scorers do not."""
