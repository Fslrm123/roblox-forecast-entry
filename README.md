# Roblox Up-and-Coming top 10 forecaster

Entry for make.fun's "Predict Roblox's Up-and-Coming top 10"
(`roblox-up-and-coming-forecast-01`).

`main.py` (Python 3.13, standard library only) reads
`/input/history.ndjson` and `/input/task.json` and writes
`/output/forecast.json`: for each of the top 10 places, a probability for
each game named being at that place, plus the forecaster's top 10 in order.

## Method

An empirical transition kernel, estimated from the history itself:

- Every pair of snapshots contributes transitions, pooled by (time gap,
  rank band) — where games at a similar rank were after a similar time gap.
- A game now at rank r is forecast by the pooled distribution of later
  ranks for its band and gap, smoothed with a Gaussian kernel over ranks,
  averaged over twelve horizons spread uniformly across the scored window.
- The forecast is blended with a persistence spike on the game's current
  rank (the current leader is stickier than the average game at its rank).
- Each place keeps 15% unnamed for games the history cannot name (new
  entrants, deep climbers). Games that will age out of the 28-day chart
  window (from the games/v1 `created` date, when the history carries
  details) have their weight scaled by the fraction of the window they can
  still be on the chart.

No network, no model calls, no randomness: two runs on the same history
write the same forecast byte for byte.

## Local check

On the kit's practice history the program scores **-34.529** on the
practice window (house baseline -43.512, house model -37.2), verified with
an independent scorer built from the competition's rules and the public
archive's reads.

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
