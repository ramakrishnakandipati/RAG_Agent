# Presentation

`Upsell_Guardrail_MAS_Deck.pptx` - 44 slides (16:9): executive summary, architecture, detailed six-stage end-to-end flow,
uplift end-to-end documentation, every model with metrics and cross-model comparisons, flow-level / stage-level / overall metrics,
and seven Streamlit-screenshot scenario stories.

Rebuild (all numbers are read from `outputs/metrics/*.json`, nothing is typed by hand):

```bash
python presentation/prep_assets.py     # crops screenshots, dumps per-example cross-model data
node presentation/build_deck.js        # needs pptxgenjs, react, react-dom, react-icons, sharp
```

After changing metrics or screenshots, rerun `python -m agents.flows`, `python scripts/verify_and_capture.py`, then the two commands above.
`assets/` holds the generated crops and `model_compare.json`.
