# WNBA Player-Impact Model + Lineup Dashboard

*Work in progress.*

A RAPM-style (Regularized Adjusted Plus-Minus) player-impact model built from real WNBA
play-by-play data, with a lineup-projection tool and a small interactive dashboard.

## Project layout

```
data/          raw and processed data (gitignored)
src/           pipeline code (data pull, possession reconstruction, model, lineup tool)
notebooks/     exploratory / validation notebooks
dashboard/     Dash app
```

## Status

- [ ] Data pull
- [ ] Lineup & possession reconstruction
- [ ] RAPM model + validation against published ratings
- [ ] Lineup projection tool
- [ ] Dashboard
- [ ] Documentation
- [ ] Deployed live demo

Full writeup (methodology, validation results, limitations) will land here once the
pipeline is built and validated.
