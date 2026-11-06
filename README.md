# ATP-predictor
ATP Match Predictor

## Setup

```bash
pipenv install
```

### Data

Match data is Jeff Sackmann's ATP dataset (CC BY-NC-SA 4.0). The original `JeffSackmann/tennis_atp`
repo was taken down in 2026, so this uses the archival mirror
[Aneeshers/tennis-sackmann-archive](https://github.com/Aneeshers/tennis-sackmann-archive)
(snapshot from June 2026, so 2026 runs through Roland Garros).

```bash
B=https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main/atp
mkdir -p data/raw
for f in $(seq -f "atp_matches_%g.csv" 2010 2026) atp_players.csv; do curl -sfL -o data/raw/$f $B/$f; done
```

Check it loads:

```bash
pipenv run python -c "import sys; sys.path.insert(0,'src'); from data_loading import load_matches; print(load_matches().shape)"
```
