# ChronoCleave AI

**Sequence models that read an embryo's developmental timeline and estimate its chance of implanting**

<p align="center">
  <img width="1760" height="496" alt="embryo_timeline" src="https://github.com/user-attachments/assets/89054dfb-0850-4855-80a4-d4722e46d29c" />
</p>

## Project brief

When several embryos are available after IVF, someone has to decide which one to transfer first. For decades that decision rested on a few snapshots: an embryologist took each dish out of the incubator once a day, looked at it for a few seconds and assigned a grade. Time lapse incubators changed what can be known. A camera inside the incubator photographs every embryo every ten minutes, so the laboratory can record the exact hour at which each cell division happened without ever disturbing the culture. Those timestamps (t2, t3, t5, the start of blastulation and so on) are called morphokinetic markers, and a large body of research shows they carry information about viability that a daily snapshot cannot see.

The practical difficulty is turning a dozen timestamps into one decision. The first generation of tools were hierarchical rules: check whether the fifth cell appeared inside a reference window, whether the second round of divisions was synchronous, whether the second cell cycle was too short, and sort embryos into classes. Rules like these are transparent, but they use three or four markers out of twelve, they treat a marker that is one minute outside its window the same as one that is ten hours outside, and they cannot be applied cleanly when an event was never annotated. Laboratories that transfer on day 3 have no blastulation timings at all, so a single rule set cannot serve both day 3 and day 5 programmes.

ChronoCleave AI treats the problem as sequence classification. Each embryo is an ordered series of developmental events with real valued timings and gaps, of variable length, with some events missing. A bidirectional LSTM and a temporal convolutional network (TCN) read that series directly, alongside static observations such as fragmentation and multinucleation, and output an implantation score. The repository contains the simulator, interval feature engineering, the rule based comparator, classical baselines, both sequence architectures, a configurable hyperparameter sweep, complete MLflow experiment tracking, a REST API that accepts JSON payloads of morphokinetic markers and returns a score with plain language flags, tests, a container and continuous integration.

Known implantation data cannot be shared, so the models are trained on 25,763 simulated transferred embryos from 16,000 patients. The simulator gives each embryo a hidden developmental competence that shapes its cell cycle regularity, division synchrony, blastulation speed and nuclear appearance, and outcome depends on that competence, on maternal age and on a patient level effect. Timings are centred on published values. Every result below describes simulated embryos and none is a clinical claim.

## What the system does

<table>
  <tr><th align="left">Capability</th><th align="left">How it is delivered</th></tr>
  <tr><td>Implantation score</td><td>TCN or bidirectional LSTM over twelve ordered events, with masked pooling so missing annotations and day 3 transfers are handled natively</td></tr>
  <tr><td>Interval features</td><td>cc2, cc3, s2, s3, t5 minus t2, compaction time, blastulation onset and duration, direct cleavage and window flags</td></tr>
  <tr><td>Comparators</td><td>Hierarchical rule score from published reference windows, logistic regression and gradient boosting on engineered intervals</td></tr>
  <tr><td>Experiment tracking</td><td>Every baseline and sweep entry is an MLflow run with parameters, per epoch metrics, test metrics and model artifacts</td></tr>
  <tr><td>Cohort ranking</td><td>Sibling embryos ranked for transfer, scored by pairwise accuracy within patient</td></tr>
  <tr><td>REST API</td><td>FastAPI service with chronological validation, percentile against the training population and clinical flags</td></tr>
</table>

## Key findings

### Learned models are far ahead of reference windows, on every view of performance

On 3,866 embryos from patients the models never saw, the hierarchical rule score reaches a ROC AUC of 0.597. Every learned model is above 0.70. The gain is just as clear on the metric closest to the real decision: when two sibling embryos from the same patient had different outcomes, the rule ranked the implanting one first 57 percent of the time, and the learned models did so 68 to 69 percent of the time.

<table>
  <tr><th align="left">Model</th><th>ROC AUC</th><th>PR AUC</th><th>Sibling ranking accuracy</th><th>ROC AUC, day 3 transfers</th><th>ROC AUC, day 5 transfers</th></tr>
  <tr><td>Rule based reference windows</td><td align="center">0.597</td><td align="center">0.335</td><td align="center">0.573</td><td align="center">0.576</td><td align="center">0.605</td></tr>
  <tr><td>Logistic regression on intervals</td><td align="center">0.710</td><td align="center">0.468</td><td align="center">0.686</td><td align="center">0.672</td><td align="center">0.721</td></tr>
  <tr><td>Gradient boosting on intervals</td><td align="center">0.714</td><td align="center">0.478</td><td align="center">0.692</td><td align="center">0.676</td><td align="center">0.725</td></tr>
  <tr><td>Bidirectional LSTM (32 units)</td><td align="center">0.708</td><td align="center">0.468</td><td align="center">0.678</td><td align="center">0.674</td><td align="center">0.717</td></tr>
  <tr><td><b>TCN (32 channels), served model</b></td><td align="center"><b>0.705</b></td><td align="center"><b>0.461</b></td><td align="center"><b>0.676</b></td><td align="center"><b>0.677</b></td><td align="center"><b>0.713</b></td></tr>
</table>

<p align="center"><img width="928" height="848" alt="roc_curves" src="https://github.com/user-attachments/assets/2ff46be7-818f-46ac-bf86-bf9f71b9a07f" /></p>
<p align="center"><img width="2080" height="640" alt="model_comparison" src="https://github.com/user-attachments/assets/8c9a978f-81f6-4dc7-8ac2-6f159d13b66b" /></p>

### Sequence models match hand built intervals without being given them

The second result is less flattering to deep learning and is reported as measured. Gradient boosting on engineered intervals scores 0.714 and the best sequence models score 0.705 to 0.708. The four learned models sit within about 0.01 of each other, which is roughly the sampling uncertainty of a test set this size, so none can be called the winner. What the comparison does show is that the LSTM and the TCN recover, from raw timestamps alone, essentially the same signal that an embryologist's interval definitions (cc2, s2, cc3 and the rest) hand to the boosted trees. The TCN is also the strongest model on day 3 transfers (0.677), where sequences are short and several engineered intervals do not exist.

For a laboratory the choice is therefore about operations, not accuracy. Interval features with boosted trees are simpler to audit. A sequence model needs no feature definitions, extends naturally if the annotation scheme adds events, and tolerates gaps without imputation. This repository serves the TCN and keeps the boosted model as a tracked comparator so the decision can be revisited on real data.

### Blastulation timing carries the most information

Hiding one event at a time from the served model and measuring the loss in test AUC shows where the signal lives. Removing the start of blastulation (tSB) costs 0.041 and removing the full blastocyst time (tB) costs 0.026. Among cleavage events, t3 matters most (0.015), because it defines both the second cell cycle and direct cleavage. Most other single events can be dropped at a cost of about 0.005 or less, because neighbouring events let the model reconstruct the interval.

<p align="center"><img width="1312" height="624" alt="event_importance" src="https://github.com/user-attachments/assets/bebfbbb3-67ad-4ffc-a201-8cb7b92a63ff" /></p>

The same point appears at cohort level. Every learned model is about 0.04 to 0.05 AUC better on day 5 transfers than on day 3 transfers. That gap is a measurable estimate of what extended culture adds to embryo selection, separate from its biological effects, and it is the kind of number a laboratory weighing day 3 against day 5 policies would want.

### The score separates embryos into clinically distinct groups

Embryos in the top quarter of TCN scores implanted 48.1 percent of the time. Those in the bottom quarter implanted 11.1 percent of the time, against an overall rate of 28.1 percent. Observed implantation rises steadily across score deciles and tracks the mean predicted score, so the output can be read as a probability and not only as a rank.

<p align="center"><img width="1120" height="640" alt="score_deciles" src="https://github.com/user-attachments/assets/066b7e29-1f2b-4e18-bc56-53b619d61506" /></p>

### What the timing profiles look like

Implanting embryos cluster inside the published t5 window (shaded), have tighter s2 values, rarely show a second cell cycle under five hours, and complete blastulation faster. The distributions overlap heavily, which is why single thresholds discriminate poorly and why combining markers helps.

<p align="center"><img width="2240" height="576" alt="timing_distributions" src="https://github.com/user-attachments/assets/019309f3-de96-4a65-885c-051483459fab" /></p>

### Every run is tracked and reproducible

One pipeline execution creates a parent MLflow run with seven nested runs. Baselines log their validation and test metrics. Sequence runs also log hyperparameters, parameter counts, loss and validation AUC at every epoch, and the trained weights. The selected run is tagged on the parent together with the exported serving bundle.

<table>
  <tr><th align="left">MLflow run</th><th align="left">Family</th><th align="left">Key parameters</th><th>Validation AUC</th><th>Test AUC</th></tr>
  <tr><td>rule_based</td><td>baseline</td><td>published windows for t5, s2, cc2</td><td align="center">0.606</td><td align="center">0.597</td></tr>
  <tr><td>logistic_regression</td><td>baseline</td><td>median imputation with indicators, C 0.5</td><td align="center">0.726</td><td align="center">0.710</td></tr>
  <tr><td>gradient_boosting</td><td>baseline</td><td>300 iterations, 15 leaves, learning rate 0.05</td><td align="center">0.730</td><td align="center">0.714</td></tr>
  <tr><td>lstm_32</td><td>sequence</td><td>1 layer, 32 units, learning rate 0.003</td><td align="center">0.720</td><td align="center">0.708</td></tr>
  <tr><td>lstm_64</td><td>sequence</td><td>2 layers, 64 units, dropout 0.2</td><td align="center">0.718</td><td align="center">0.707</td></tr>
  <tr><td>tcn_32 (selected)</td><td>sequence</td><td>3 levels, 32 channels, kernel 3</td><td align="center">0.721</td><td align="center">0.705</td></tr>
  <tr><td>tcn_64</td><td>sequence</td><td>3 levels, 64 channels, dropout 0.2</td><td align="center">0.720</td><td align="center">0.705</td></tr>
</table>

<p align="center"><img width="1120" height="624" alt="learning_curves" src="https://github.com/user-attachments/assets/151ec640-8a70-4777-8deb-0ea728f975a9" /></p>

Selection among sequence models uses validation AUC only. Test metrics are logged for transparency and are never used to choose a model. Run `make ui` to browse the runs in the MLflow interface.

## Architecture

```
time lapse annotations (CSV or JSON)
        |
        v
grouped split by patient          sibling embryos never cross a split boundary
        |
        +==> interval features ==> rule score, logistic regression, gradient boosting
        |
        +==> event sequence encoding (time, gap since last annotated event, observed flag, event embedding)
                    |
                    +==> bidirectional LSTM  }  masked mean and max pooling
                    +==> dilated TCN         }  joined with static covariates, then a dense head
        |
        v
MLflow experiment (parent run, nested baseline and sweep runs, artifacts)
        |
        v
serving bundle ==> FastAPI  /score  /rank  /model  /health
```

## Repository layout

```
chronocleave_ai
    configs/config.yaml                 data, sweep definition, MLflow settings
    artifacts/chronocleave_bundle.pt    selected model, scaler and score quantiles
    docs/images                         figures in this document
    reports/metrics.json                all results, histories and occlusion importances
    src/chronocleave
        data/schema.py                  event vocabulary and reference windows
        data/simulate.py                known implantation data simulator
        data/prepare.py                 patient grouped split and sequence encoding
        features/intervals.py           interval features, rule score, clinical flags
        models/sequence.py              LSTM and TCN classifiers
        training/sequence_trainer.py    training loop with early stopping
        training/pipeline.py            baselines, sweep, MLflow logging, export
        evaluation/metrics.py           AUC, sibling ranking accuracy, quartile lift
        evaluation/report.py            figure generation
        inference/scorer.py             validation, scoring, ranking
        api/main.py                     FastAPI service
    tests                               18 tests including a full pipeline run with a private MLflow store
    Dockerfile, compose.yaml            API container and an MLflow interface container
```

## Getting started

Python 3.10 or later is required.

```
make install
make all
make test
```

`make all` simulates the cohort, runs the baselines and the four sweep entries, logs everything to MLflow and rebuilds the figures. It takes a few minutes on a laptop CPU. A trained bundle is included, so the API works straight after installation.

```
make api
make ui
```

The API documentation is at `http://localhost:8000/docs` and the MLflow interface at `http://localhost:5000`.

### Calling the API

```python
import httpx

embryo = {
    "embryo_id": "E1", "tPNf": 23.1, "t2": 25.6, "t3": 36.9, "t4": 37.4, "t5": 50.2, "t6": 51.5, "t7": 52.8, "t8": 54.0,
    "tM": 81.5, "tSB": 93.0, "tB": 101.2, "tEB": 108.9, "maternal_age": 33, "fragmentation_pct": 5,
}
print(httpx.post("http://localhost:8000/score", json=embryo).json())

sibling = dict(embryo, embryo_id="E2", t3=28.0)   # second cell cycle of 2.4 hours: direct cleavage
ranked = httpx.post("http://localhost:8000/rank", json={"embryos": [embryo, sibling]}).json()
for item in ranked["embryos"]:
    print(item["rank"], item["embryo_id"], item["implantation_score"], item["flags"])
```

Events that were not annotated are simply left out; only t2 is required. Timings that are out of chronological order or outside 0 to 160 hours are rejected with a message naming the offending event. Each response includes the derived intervals and any flags, for example a direct cleavage or a t5 outside its reference window, so the score never arrives without context.

### Adding a sweep entry

Add an entry such as `{model: tcn, channels: 96, levels: 4, kernel: 3, dropout: 0.2, learning_rate: 0.001}` under `training.sweep` in the configuration file and run `make train`. Each entry becomes its own MLflow run.

## Limitations and responsible use

* The data is simulated. Real morphokinetic timings vary with culture media, oxygen tension, fertilisation method and annotation practice, and a model must be validated on a laboratory's own known implantation data.
* Only transferred embryos have outcomes. This selection effect exists in real data too and limits what any model can learn about embryos that were never chosen.
* The model ranks embryos; it does not assess ploidy and is not a substitute for embryologist review.
* ChronoCleave AI is a research and portfolio prototype. It is not a medical device.

## Licence

Released under the MIT licence.
