# UNIFY-RUL — Core Course Project Report

> Master text for the submission document. Structure, chapter order, table and
> figure numbering follow the department template used in
> `24EC0215_and_24EC0247(CCP_2ND_YEAR).pdf`. `scripts/make_docx_report.py`
> renders this into the formatted `.docx`. Every measured number traces to
> `reports/full-test-2026-08-28.md`.

---

## TITLE PAGE

::: center

**UNIFY-RUL: UNIVERSAL REMAINING USEFUL LIFE PREDICTION FOR**

**PREDICTIVE MAINTENANCE — ANY DATASET, ANY HARDWARE**

A PROJECT REPORT

Submitted by

VISHNU VARDHAN K S

(24EC0251)

VS KAMLESH

(24EC0243)

This project report is submitted as part of the requirements for completion.

of

CORE COURSE PROJECT OF THE CURRICULUM

*[LOGO]*

DEPARTMENT OF ELECTRONICS AND COMMUNICATION ENGINEERING

CHENNAI INSTITUTE OF TECHNOLOGY

(An Autonomous Institution, Affiliated to Anna University, Chennai)

SARATHY NAGAR, KUNDRATHUR – 600 069

TAMIL NADU, INDIA

AUGUST 2026

:::

---

## BONAFIDE CERTIFICATE PAGE

::: center

CHENNAI INSTITUTE OF TECHNOLOGY

(An Autonomous Institution, Affiliated to Anna University, Chennai)

SARATHY NAGAR, KUNDRATHUR – 600 069

*[LOGO]*

**BONAFIDE CERTIFICATE**

:::

Certified that this project report entitled "UNIFY-RUL: UNIVERSAL REMAINING
USEFUL LIFE PREDICTION FOR PREDICTIVE MAINTENANCE — ANY DATASET, ANY HARDWARE"
is the Bonafide work of Mr. VISHNU VARDHAN K S (Reg. No.: 24EC0251),
Mr. VS KAMLESH (Reg. No.: 24EC0243) who carried out the research under our
supervision.

::: plain

| Dr. ARULANANTH T S | Mr. PREM SANGEETH |
|---|---|
| Head of the Department | |
| Professor, | Associate Professor, |
| Department of Electronics and | Department of Electronics and |
| communication Engineering, | communication Engineering, |
| Chennai Institute of Technology, | Chennai Institute of Technology, |
| Kundrathur – 600 069. | Kundrathur – 600 069. |

:::

Certified that the student have attended viva voce during the exam held on
…………………….

::: plain

| INTERNAL EXAMINAR | EXTERNAL EXAMINAR |
|---|---|

:::

---

## INSTITUTION VISION AND MISSION

*[LETTERHEAD]*

**1. VISION OF THE INSTITUTION:**

::: box-purple

To be an eminent centre for Academia, Industry and Research by imparting knowledge, relevant practices and inculcating human values to address global challenges through novelty and sustainability.

:::

**2. MISSION OF THE INSTITUTION:**

::: box-purple

**IM1:** To create next generation leaders by effective teaching learning methodologies and instill scientific spark in them to meet the global challenges.

**IM2:** To transform lives through deployment of emerging technology, novelty and sustainability.

**IM3:** To inculcate human values and ethical principles to cater to the societal needs.

**IM4:** To contribute towards the research ecosystem by providing a suitable, effective platform for interaction between industry, academia and R & D establishments.

**IM5:** To nurture incubation centres enabling structured entrepreneurship and start-ups.

:::

---

## DEPARTMENT VISION AND MISSION

*[LETTERHEAD]*

::: center

**Department of Electronics and communication Engineering**

:::

**VISION OF THE DEPARTMENT:**

::: box-green

To Excel in the emerging areas of Electronics and Communication Engineering by imparting knowledge, relevant practices and inculcating human values to transform the students as potential resources to cater the industrial and societal development through sustainable technology growth.

:::

**MISSION OF THE DEPARTMENT:**

::: box-green

**DM1:** To provide strong fundamentals and technical skills through effective teaching learning Methodologies.

**DM2:** To transform lives of the students by fostering ethical values, creativity and innovation to become Entrepreneurs and establish Start-ups.

**DM3:** To habituate the students to focus on sustainable solutions to improve the quality of life and welfare of the society.

**DM4:** To provide an ambience for research through collaborations with industry and academia.

**DM5:** To inculcate learning of emerging technologies for pursuing higher studies leading to lifelong learning.

:::

---

## PROGRAM OUTCOMES

*[LETTERHEAD]*

**Program Outcomes as defined by NBA (PO) Engineering Graduates will be able to:**

::: compact

1. **Engineering knowledge:** Apply the knowledge of mathematics, science, engineering fundamentals, and an engineering specialization to the solution of complex engineering problems.

2. **Problem analysis:** Identify, formulate, review research literature, and analyze complex engineering problems reaching substantiated conclusions using first principles of mathematics, natural sciences, and engineering sciences.

3. **Design/development of solutions:** Design solutions for complex engineering problems and design system components or processes that meet the specified needs with appropriate consideration for the public health and safety, and the cultural, societal, and environmental considerations.

4. **Conduct investigations of complex problems:** Use research-based knowledge and research methods including design of experiments, analysis and interpretation of data, and synthesis of the information to provide valid conclusions.

5. **Modern tool usage:** Create, select, and apply appropriate techniques, resources, and modern engineering and IT tools including prediction and modeling to complex engineering activities with an understanding of the limitations.

6. **The engineer and society:** Apply reasoning informed by the contextual knowledge to assess societal, health, safety, legal and cultural issues and the consequent responsibilities relevant to the professional engineering practice.

7. **Environment and sustainability:** Understand the impact of the professional engineering solutions in societal and environmental contexts, and demonstrate the knowledge of, and need for sustainable development.

8. **Ethics:** Apply ethical principles and commit to professional ethics and responsibilities and norms of the engineering practice.

9. **Individual and team work:** Function effectively as an individual, and as a member or leader in diverse teams, and in multidisciplinary settings.

10. **Communication:** Communicate effectively on complex engineering activities with the engineering community and with society at large, such as, being able to comprehend and write effective reports and design documentation, make effective presentations, and give and receive clear instructions.

11. **Project management and finance:** Demonstrate knowledge and understanding of the engineering and management principles and apply these to one's own work, as a member and leader in a team, to manage projects and in multidisciplinary environments.

12. **Life-long learning:** Recognize the need for, and have the preparation and ability to engage in independent and life-long learning in the broadest context of technological change.

:::

---

## PROGRAM SPECIFIC OUTCOMES AND OBJECTIVES

*[LETTERHEAD]*

**Program Specific Outcomes (PSOs)**

| S.No. | Programme Specific Outcomes |
|---|---|
| PSO1 | To analyze, design and develop quality solutions in Communication Engineering by adapting the emerging technologies. |
| PSO2 | To innovate ideas and solutions for real time problems in industrial and domestic Automation using Embedded & IOT tools. |

**Program Educational Objectives:**

::: box-green

Graduates will be able to

**PEO1:** Contribute to the industry as an Engineer through sound knowledge acquired in core engineering to develop new processes and implement the solutions for industrial problems.

**PEO2:** Establish an organization / industry as an Entrepreneur with professionalism, leadership quality, teamwork, and ethical values to meet the societal needs.

**PEO3:** Create a better future by pursuing higher education / research and develop the sustainable products / solutions to meet the demand.

:::

---

## ACKNOWLEDGEMENT

It is our first and foremost duty to express our heartily gratitude and
allegiance to our beloved chairman Sri. P. SRIRAM for providing us a good
working environment, facilities and opportunity to carry out the research work
with full freedom in and out of the campus.

We would like to express our deepest gratitude to our Principal Dr. A. RAMESH,
Ph.D., for being our inspiration throughout the entire period of the course.

We greatly appreciate and convey our heartfelt thanks to Dr. ARULANANTH,
M.Tech, Ph.D., Professor & Head of Department – Electronics and Communication
Engineering, for their continuous support and valuable suggestions throughout
the research period.

It gives us an immense pleasure to express our deep sense of gratitude to our
research supervisor, Mr. PREM SANGEETH, M.E, Ph.D., Associate Professor,
Faculty of Electronics and Communication Engineering for their valuable
guidance, constant inspiration, encouragement, and motivation throughout the
course of this project.

We register our gratitude and sincere thanks to all the faculty and supporting
staff of the Department of Electronics and Communication Engineering for their
valuable suggestions and support during the course of our research work.

We thank ALMIGHTY to have provided such great personalities in our life and the
chance to join their research group.

VISHNU VARDHAN K S (24EC0251)
VS KAMLESH (24EC0243)

---

## TABLE OF CONTENTS

| CHAPTER NO. | TITLE | PAGE NO. |
|---|---|---|
| | Abstract | 3 |
| | Acknowledgement | 4 |
| | List of Tables | 6 |
| | List of Figures | 7 |
| | List of Symbols and Abbreviations | 8 |
| **1** | **INTRODUCTION** | 9 |
| 1.1 | Background and Motivation | 9 |
| 1.2 | Problem Statement | 11 |
| 1.3 | Objectives of the Project | 12 |
| 1.4 | Scope of the Work | 13 |
| 1.5 | Organization of the Report | 14 |
| **2** | **LITERATURE REVIEW** | 15 |
| 2.1 | Remaining Useful Life Estimation – Prior Work | 15 |
| 2.2 | Run-to-Failure Datasets and Benchmarks | 17 |
| 2.3 | Edge AI and Model Compression Techniques | 18 |
| 2.4 | Attention Mechanisms for Multivariate Sensor Data | 19 |
| 2.5 | Research Gaps and Motivation | 21 |
| **3** | **SYSTEM DESIGN AND METHODOLOGY** | 22 |
| 3.1 | Design Overview | 22 |
| 3.2 | Dataset Model – C-MAPSS Turbofan Degradation | 23 |
| 3.3 | Preprocessing and Per-Condition Normalisation | 24 |
| 3.4 | Sensor-Agnostic Model Architecture | 25 |
| 3.5 | Piecewise RUL Labelling and Loss Function | 26 |
| 3.6 | Alarm Logic and Hysteresis | 27 |
| **4** | **IMPLEMENTATION** | 28 |
| 4.1 | Training Pipeline Implementation | 28 |
| 4.2 | ONNX Export Implementation | 29 |
| 4.3 | INT8 Quantisation | 30 |
| 4.4 | Edge Inference Service Implementation | 31 |
| 4.5 | Sensor Node and Replay Implementation | 32 |
| **5** | **EXPERIMENTS AND RESULTS** | 33 |
| 5.1 | Experimental Setup and Environment | 33 |
| 5.2 | Training Convergence | 34 |
| 5.3 | Prediction Accuracy Analysis | 35 |
| 5.4 | Comparative Results: Proposed Model vs Classical Baseline | 37 |
| 5.5 | Edge Deployment Performance | 38 |
| 5.6 | Live Run-to-Failure Validation | 40 |
| **6** | **CONCLUSION AND FUTURE SCOPE** | 42 |
| 6.1 | Conclusion | 42 |
| 6.2 | Future Scope | 43 |
| **7** | **APPENDICES** | 45 |
| **8** | **REFERENCES** | 47 |

---

## LIST OF TABLES

| TABLE No. | TITLE | PAGE No. |
|---|---|---|
| 3.1 | C-MAPSS Subset Characteristics | 23 |
| 3.2 | Model Architecture Parameters | 25 |
| 3.3 | Training Hyperparameters | 26 |
| 4.1 | ONNX Export Verification Across Sensor Counts | 29 |
| 4.2 | Deployment Footprint After Quantisation | 30 |
| 4.3 | Edge Service Interface Summary | 31 |
| 5.1 | Experimental Environment Parameters | 33 |
| 5.2 | Accuracy Comparison – Proposed Model vs RandomForest | 37 |
| 5.3 | Inference Latency Across Thread Counts | 38 |
| 5.4 | Post-Export and Post-Quantisation Accuracy | 39 |
| 5.5 | Performance Summary Table | 41 |

## LIST OF FIGURES

| FIGURE No. | TITLE | PAGE No. |
|---|---|---|
| 1.1 | Maintenance Strategy Comparison – Reactive, Preventive, Predictive | 10 |
| 1.2 | Degradation Trajectory and Remaining Useful Life | 11 |
| 2.1 | Conventional Cloud-Centric RUL Prediction Pipeline | 16 |
| 3.1 | Proposed System Architecture Block Diagram | 22 |
| 3.2 | End-to-End Data Flow Flowchart | 23 |
| 3.3 | Piecewise Linear RUL Labelling with Cap at 125 Cycles | 26 |
| 4.1 | Deployment Footprint – Checkpoint, FP32 ONNX, INT8 ONNX | 30 |
| 5.1 | Training Convergence – Validation RMSE and Training Loss | 34 |
| 5.2 | Predicted vs True RUL on the Official Test Split | 35 |
| 5.3 | Error Distribution on the Official Test Split | 36 |
| 5.4 | Proposed Model vs RandomForest Baseline | 37 |
| 5.5 | Inference Latency Sweep on Raspberry Pi 5 | 38 |
| 5.6 | Inference Cost Against the Real-Time Budget | 39 |
| 5.7 | Live Run-to-Failure Trace on the Raspberry Pi 5 | 40 |
| 5.8 | End-to-End Service Latency Distribution | 41 |
| 5.9 | Edge Dashboard Served from the Raspberry Pi 5 | 42 |

## LIST OF SYMBOLS, ABBREVIATIONS AND NOMENCLATURE

| SYMBOL / ABBREVIATION | DESCRIPTION |
|---|---|
| RUL | Remaining Useful Life |
| PdM | Predictive Maintenance |
| C-MAPSS | Commercial Modular Aero-Propulsion System Simulation |
| FD001–FD004 | The four C-MAPSS operating-condition subsets |
| FEMTO | FEMTO-ST bearing run-to-failure dataset |
| XJTU-SY | Xi'an Jiaotong University bearing degradation dataset |
| RMSE | Root Mean Square Error |
| MAE | Mean Absolute Error |
| NASA Score | Asymmetric C-MAPSS scoring function penalising late predictions |
| ONNX | Open Neural Network Exchange |
| ORT | ONNX Runtime |
| INT8 | 8-bit integer quantised representation |
| FP32 | 32-bit floating point representation |
| CNN | Convolutional Neural Network |
| LSTM | Long Short-Term Memory network |
| GRU | Gated Recurrent Unit |
| EWMA | Exponentially Weighted Moving Average |
| SoC | System on Chip |
| API | Application Programming Interface |
| REST | Representational State Transfer |
| SBC | Single Board Computer |
| MCU | Microcontroller Unit |
| p50 / p95 / p99 | 50th / 95th / 99th percentile |
| L | Window length in timesteps |
| S | Number of sensor channels in a window |
| ŷ | Predicted normalised life fraction, in [0, 1] |
| y | True normalised life fraction, in [0, 1] |
| σ | Standard deviation used for channel normalisation |
| μ | Mean used for channel normalisation |
| α | EWMA smoothing coefficient |

---

## ABSTRACT

Industrial machinery fails, and the cost of a failure is dominated not by the
part that broke but by the unplanned downtime around it. Predictive maintenance
answers this by estimating the Remaining Useful Life (RUL) of an asset from its
sensor history, so that service can be scheduled before failure rather than
after it. Two obstacles keep such systems out of ordinary industrial practice.
First, published RUL models are written against one dataset with one fixed
sensor layout, and adding, removing or reordering a sensor requires retraining a
new architecture. Second, those models are sized for cloud inference, whereas a
factory floor needs a prediction on a small, cheap, network-independent board
next to the machine.

This project presents UNIFY-RUL, a compact attention-based RUL prediction system
designed around both constraints. The model treats each sensor channel as a
token carrying a learned identity embedding, so the sensor axis is a variable
rather than part of the architecture: one set of weights accepts an arbitrary
number of channels in an arbitrary order, and missing channels are handled by an
attention mask instead of retraining. The network is deliberately small at 44,417
parameters, and predicts a normalised fraction of life in the interval [0, 1]
which is denormalised per dataset, so that datasets recorded in cycles, seconds
and minutes can share one output head without the largest-numbered dataset
dominating the loss.

The system was trained on the NASA C-MAPSS turbofan degradation benchmark, using
all four operating-condition subsets — 1,416 engine units and 265,256 timesteps
— with the published `test_FDxxx` files held out and never used for training or
model selection. On that official test split the model attains an RMSE of 14.895
cycles, an MAE of 10.188 cycles and a NASA score of 3,831 over 690 engines, with
72.5 % of engines predicted within ±13 cycles of their true remaining life. A
RandomForest regressor over 105 engineered features was trained as a classical
reference and reaches 14.405 RMSE, marginally better on accuracy — a result this
report presents plainly, because the case for the proposed architecture rests on
deployability rather than on leaderboard position.

The trained network is exported to ONNX with a dynamic sensor axis and verified
to reproduce the PyTorch model to within 1.19 × 10⁻⁷ across sensor counts from 2
to 21, and INT8 quantisation reduces the graph from 190.4 KB to 79.3 KB at a cost
of 0.044 cycles RMSE, or 0.29 %. The complete inference service was deployed to a
Raspberry Pi 5 running CPU-only ONNX Runtime, where a single 30-cycle window is
predicted in 0.654 ms at the median and 0.675 ms at the 95th percentile,
consuming 0.068 % of a one-second sensing budget and leaving roughly 1,481× of
headroom. End-to-end latency through the REST service, including windowing,
normalisation and persistence, is 1.19 ms at the median.

Final validation streamed a real C-MAPSS engine through the deployed service one
cycle at a time from health to failure. The system tracked the true degradation
trajectory, predicted an RUL of 0.0 at the true end of life, and raised its alarm
32 cycles before failure with zero false alarms during the 137 healthy cycles
that preceded it. This live test also exposed a deployment defect invisible to
offline evaluation — a normalisation statistics mismatch that produced a
confident but badly wrong prediction — which was diagnosed, fixed at its root and
regression-tested, and is documented in this report as a result in its own right.

---

## CHAPTER 1  INTRODUCTION

### 1.1 Background and Motivation

Maintenance of industrial machinery has historically followed one of two
strategies, and both waste money in opposite directions. Reactive maintenance
runs an asset until it breaks and then repairs it; the repair is cheap to
schedule but the unplanned downtime around it is not, and a failure that occurs
mid-production can idle an entire line. Preventive maintenance replaces
components on a fixed calendar or running-hours schedule; downtime becomes
predictable, but perfectly serviceable parts are discarded, and a component that
degrades faster than the schedule assumed still fails in service.

Predictive maintenance is the third strategy. Instead of guessing from a
calendar, it estimates the condition of the asset from the sensors already
attached to it — vibration, temperature, pressure, current, speed — and
schedules service just before the estimated point of failure. The central
quantity in this strategy is the Remaining Useful Life: the amount of operating
time an asset has left before it can no longer perform its function. A system
that estimates RUL accurately converts an unplanned stoppage into a planned one,
which is where essentially all of the economic value of predictive maintenance
sits.

*[Figure 1.1: Maintenance Strategy Comparison – Reactive, Preventive, Predictive]*

The difficulty is that RUL is not directly measurable. It must be inferred from
sensor signals whose relationship to the underlying damage is indirect,
nonlinear, and contaminated by changes in operating condition that have nothing
to do with degradation. A jet engine climbing at altitude and the same engine
cruising produce very different sensor readings while being equally healthy. A
useful RUL model must separate the part of the signal that encodes damage from
the part that merely encodes what the machine is currently doing.

*[Figure 1.2: Degradation Trajectory and Remaining Useful Life]*

The last decade of research has answered this with deep learning, and answered
it well: convolutional and recurrent networks trained on run-to-failure datasets
now predict RUL with errors small enough to be operationally useful. But the
research answer and the deployable answer have diverged in two respects that
motivate this project.

The first is architectural rigidity. A published RUL model is written against
one dataset. Its input layer is shaped by that dataset's sensor count, its
normalisation constants are that dataset's, and its output is in that dataset's
unit of time. Deploying it on a different machine — with eleven sensors instead
of fifteen, in a different order, one of them failed — means designing and
training a new model. For a research paper this is irrelevant; for a maintenance
department with two hundred heterogeneous assets it is the entire problem.

The second is deployment weight. Sending raw high-rate sensor data to a cloud
service for inference assumes continuous connectivity and accepts network
latency, and neither assumption survives contact with a real factory floor,
a wind farm, or a moving vehicle. Inference belongs next to the machine, on
hardware costing tens of dollars and drawing a few watts. That is a hard
constraint on model size and on the inference runtime, and it is not a
constraint most published architectures were designed against.

This project is motivated by the observation that these two problems have a
shared solution. If the sensor axis is treated as data rather than as
architecture — each channel carrying a learned identity, combined by attention
rather than by a fixed-width layer — then the same weights serve any sensor
layout, and the model that results is naturally small, because it learns one
shared representation of degradation rather than a separate pathway per channel.
A model small enough to be sensor-agnostic turns out to be small enough to
deploy on a single-board computer, and the two goals reinforce each other rather
than competing.

### 1.2 Problem Statement

Conventional deep-learning RUL prediction systems are difficult to deploy in
real maintenance settings for the following specific reasons:

(i) The input dimensionality is fixed at design time. A network trained on a
15-sensor asset cannot accept an 11-sensor asset without architectural change
and full retraining, so every distinct machine type in a plant requires its own
model, its own training run, and its own maintenance.

(ii) Sensor failure is treated as a data problem rather than a runtime
condition. When a channel drops out in service, systems that expect a fixed
input either crash, or silently impute a value and continue predicting with no
indication that they are extrapolating.

(iii) Run-to-failure datasets record time in incompatible units — C-MAPSS in
flight cycles, FEMTO-ST in seconds, XJTU-SY in minutes — so a model trained
jointly across them optimises predominantly for whichever dataset has the
largest raw numbers, and the reported joint error is dominated by that dataset
rather than describing all of them.

(iv) Models sized for GPU inference cannot meet the latency, memory and power
budgets of an edge node, and the conversion path from a research checkpoint to
an embedded runtime is treated as an afterthought, so the accuracy that is
reported is not the accuracy that is deployed.

(v) Offline validation is accepted as proof of correctness. A model evaluated
only on cached test arrays can pass every offline metric while the deployed
service around it — its windowing, its normalisation, its configuration — is
wrong, and the wrongness is invisible because the system reports a confident
number rather than an error.

### 1.3 Objectives of the Project

The primary objectives of this project are as follows:

(i) To design a RUL prediction model whose sensor axis is dynamic rather than
architectural, using per-channel identity embeddings and attention pooling so
that one set of trained weights accepts an arbitrary number of channels in an
arbitrary order, and to verify this property across sensor counts from 2 to 21.

(ii) To make the model robust to missing channels at inference time through an
attention mask, so that a sensor which fails in service degrades the prediction
rather than breaking the service.

(iii) To predict a normalised fraction of life in the interval [0, 1] and
denormalise per dataset through a stored life scale, so that datasets recorded
in different units of time can share one model and one output head.

(iv) To constrain the model to a size deployable on a low-cost CPU-only
single-board computer, and to demonstrate inference latency at least two orders
of magnitude below the sensing period.

(v) To export the trained model to ONNX with a dynamic sensor axis, to quantise
it to INT8, and to re-measure accuracy after each conversion step so that the
reported accuracy is the accuracy of the artifact that actually ships.

(vi) To implement a complete edge inference service — ingest, windowing,
normalisation, prediction, alarm hysteresis and persistence — and to deploy and
measure it on a Raspberry Pi 5.

(vii) To validate the deployed system end to end by streaming a real
run-to-failure engine through it one cycle at a time, and to report the
prediction trajectory, the alarm lead time and the false-alarm count.

(viii) To benchmark the proposed model against a classical machine-learning
baseline on the same official test split, and to report the comparison honestly
whichever way it falls.

### 1.4 Scope of the Work

The scope of this project encompasses the following activities: design of the
sensor-agnostic attention architecture; implementation of the parsing,
windowing, per-operating-condition normalisation and piecewise RUL labelling
pipeline for the NASA C-MAPSS benchmark; joint training with dataset-balanced
sampling; evaluation on the published C-MAPSS test split using RMSE, MAE and the
asymmetric NASA score; training of a RandomForest baseline over engineered
features for comparison; ONNX export with a verified dynamic sensor axis; INT8
quantisation with post-quantisation accuracy re-measurement; implementation of a
REST inference service with windowing, EWMA alarm hysteresis and SQLite
persistence; deployment to a Raspberry Pi 5; latency benchmarking across thread
counts with thermal monitoring; and end-to-end validation by replaying a real
run-to-failure engine through the deployed service.

The scope does not include a physical instrumented test rig, training on the
FEMTO-ST and XJTU-SY bearing datasets, federated or on-device training, power
consumption measurement in watts, or a multi-node deployment. Each of these is
identified as a natural extension in the future scope section.

### 1.5 Organization of the Report

This report is organized into six chapters.

Chapter 1 provides an introduction to predictive maintenance and remaining
useful life estimation, the motivation for sensor-agnostic and edge-deployable
prediction, the problem statement, and the objectives of the project.

Chapter 2 presents a literature review of existing work on RUL estimation,
run-to-failure benchmark datasets, edge AI and model compression techniques, and
attention mechanisms applied to multivariate sensor data, concluding with the
research gaps that motivate this work.

Chapter 3 describes the proposed system design and methodology, including the
overall architecture, the C-MAPSS dataset model, per-operating-condition
normalisation, the sensor-agnostic encoder, piecewise RUL labelling, and the
alarm logic.

Chapter 4 details the implementation, including the training pipeline, the ONNX
export path and its verification, INT8 quantisation, the edge inference service,
and the sensor node and replay tooling.

Chapter 5 presents the experimental setup, training convergence, prediction
accuracy on the official test split, comparison against the classical baseline,
edge deployment performance on the Raspberry Pi 5, and the live run-to-failure
validation including the deployment defect that validation exposed.

Chapter 6 concludes the report with a summary of contributions, key findings,
and potential directions for future work.

---

## CHAPTER 2  LITERATURE REVIEW

### 2.1 Remaining Useful Life Estimation – Prior Work

Prognostics research divides RUL estimation into model-based, data-driven and
hybrid approaches. Model-based methods describe degradation with an explicit
physical or statistical law — a Paris-law crack growth model, a Wiener process
with drift, a particle filter over a state-space degradation model — and
estimate RUL by extrapolating the fitted state to a failure threshold. These
methods are interpretable and require little data, but they demand a degradation
physics that is known and stationary, which is rarely true of a complete
industrial asset.

Data-driven methods learn the mapping from sensor history to remaining life
directly. Early work applied multilayer perceptrons and support vector
regression over hand-engineered features. The decisive shift came with deep
sequence models: convolutional networks applied across the time axis of a sliding
window learn local degradation signatures without feature engineering, and
recurrent architectures — LSTM and GRU — model the longer-range temporal
dependence of a degradation trajectory. Reported RMSE on the C-MAPSS FD001
subset moved from the high twenties for classical regressors into the twelve to
sixteen range for deep sequence models, and subsequent transformer-based and
hybrid CNN-LSTM designs have continued to narrow that band.

Two practices from this literature are adopted directly in the present work. The
first is the piecewise linear RUL target: since sensor readings are essentially
flat while an asset is healthy, labelling early cycles with their true large
remaining life forces the model to distinguish states that are not
distinguishable, and capping the target at a constant during the healthy phase
consistently improves reported error. The second is the asymmetric NASA scoring
function, which penalises a late prediction — one that promises more remaining
life than exists — far more heavily than an early one, correctly reflecting that
in maintenance the two error directions have very different costs.

*[Figure 2.1: Conventional Cloud-Centric RUL Prediction Pipeline]*

### 2.2 Run-to-Failure Datasets and Benchmarks

Progress in this field is anchored on a small number of public run-to-failure
datasets, because degradation data is expensive: it can only be produced by
destroying the asset.

The NASA C-MAPSS turbofan degradation dataset remains the field's standard
benchmark. It comprises four subsets, FD001 through FD004, generated by a
high-fidelity engine simulation under one or six operating conditions and with
one or two concurrent fault modes. Each subset ships a training file in which
every engine runs to failure, a test file in which every engine is truncated at
some point before failure, and a file of true RUL values for the test engines.
Because the test split is published and fixed, results across papers are
directly comparable — provided authors evaluate on it rather than on a private
split, a distinction this project takes care to preserve.

The FEMTO-ST bearing dataset, used in the IEEE PHM 2012 challenge, records
accelerated bearing degradation with horizontal and vertical accelerometers
sampled at 25.6 kHz. The XJTU-SY dataset provides fifteen run-to-failure bearings
across three operating conditions with a similar acquisition setup. Both record
time in units incompatible with C-MAPSS, and both have far fewer units, which
makes them a strong test of transfer but a weak test of accuracy in isolation.

The heterogeneity across these three benchmarks — different sensor counts,
different sampling rates, different units of time, different failure physics —
is precisely the heterogeneity a deployable system must absorb, which is why
they are the intended joint training target of this architecture.

### 2.3 Edge AI and Model Compression Techniques

Edge AI moves inference from a data centre to hardware physically near the
sensor. The motivation is latency, bandwidth, availability and privacy: a
prediction computed locally does not wait for a network round trip, does not
require raw signals to leave the site, and does not stop when connectivity does.

The literature on making models fit such hardware is mature. Quantisation
represents weights and activations in reduced precision, most commonly 8-bit
integers, typically reducing model size by around four times and accelerating
inference on integer hardware, at a small accuracy cost that must be measured
rather than assumed. Pruning removes weights or entire channels whose
contribution is negligible. Knowledge distillation trains a small student
network to reproduce the outputs of a larger teacher. Neural architecture search
optimises the topology directly against a latency or size budget.

On the runtime side, ONNX has become the practical interchange format between
training frameworks and embedded inference engines, with ONNX Runtime,
TensorFlow Lite and TensorRT the common execution targets. The literature is
consistent on one point that this project takes seriously: a compression claim
is only meaningful when the accuracy of the compressed artifact is re-measured
on the same held-out data, because the conversion path itself can change
numerical behaviour independently of the intended compression.

### 2.4 Attention Mechanisms for Multivariate Sensor Data

Attention computes a weighted combination of a set of elements, where the
weights are produced from the content of the elements themselves. Applied along
the time axis, attention lets a sequence model reference distant history without
the vanishing-gradient limitations of recurrence, and transformer variants of
this idea now dominate sequence modelling generally.

Less commonly exploited, and central to this project, is attention along the
*sensor* axis. If each channel is embedded as a token — its raw window encoded
into a vector and summed with a learned embedding of the channel's identity —
then a set of channels can be pooled by attention into a fixed-size
representation regardless of how many channels there are. The architecture
becomes independent of the sensor count, the attention weights become a readable
statement of which sensors the model is currently relying on, and a channel that
is absent is handled by masking it out of the attention softmax rather than by
substituting a value.

This treatment of the sensor axis as a set rather than as a fixed vector is well
established in the point-cloud and set-transformer literature but is applied only
sparsely to industrial prognostics, where fixed-layout assumptions remain the
norm. The interpretability that comes with it is a practical bonus in a
maintenance context: an engineer can ask which sensors drove an alarm and receive
a per-channel weight rather than an opaque score.

### 2.5 Research Gaps and Motivation

Three gaps emerge from the reviewed literature.

Firstly, almost all reported RUL architectures are dataset-specific by
construction. Their input layers encode a sensor count, their preprocessing
encodes a sensor order, and their outputs encode a unit of time. Published work
reports accuracy on one benchmark and does not address what happens when the
sensor layout changes, even though changing sensor layouts are the normal
condition of a real maintenance fleet rather than an edge case. Cross-dataset
generality is discussed as future work far more often than it is demonstrated.

Secondly, edge deployment is treated as an implementation detail appended to a
modelling contribution rather than as a design constraint that shapes it. Papers
that report a compression ratio frequently do not report the post-compression
accuracy on the same test split, and papers that report edge latency frequently
measure it on a workstation CPU rather than on the target board. The result is
that the deployability claims in the literature are systematically weaker than
the accuracy claims beside them.

Thirdly, validation almost universally stops at offline evaluation over cached
arrays. The serving path — the code that windows the live stream, looks up
normalisation constants, and turns a model output into an alarm — is where
deployed prognostic systems actually fail, and it is exercised by no offline
metric. A system can be correct as a model and wrong as a service, and nothing
in the standard evaluation protocol would reveal it.

The proposed project is motivated by the need to close these three gaps
together. It treats the sensor axis as data, sizes the model for the target
board from the outset, re-measures accuracy after every conversion step, and
validates the complete deployed service by streaming a real engine through it
from health to failure. The third of these turned out to be the most valuable:
it exposed a serving-path defect that every offline metric had passed, which is
reported in Section 5.6 as a primary result of this work rather than as an
appendix note.

---

## CHAPTER 3  SYSTEM DESIGN AND METHODOLOGY

### 3.1 Design Overview

The proposed UNIFY-RUL system consists of six primary functional modules
operating as a pipeline: (1) the dataset parsers, which convert heterogeneous
raw formats into a common unit representation; (2) the normaliser, which fits
per-channel and per-operating-condition statistics on training units only;
(3) the windowing and labelling module, which produces fixed-length windows with
piecewise RUL targets; (4) the sensor-agnostic encoder and attention pooling
model; (5) the export and quantisation stage, which converts the trained model
into a verified ONNX artifact; and (6) the edge inference service, which
ingests live packets, maintains the window, predicts, applies alarm hysteresis
and persists the result.

The design philosophy is that the sensor axis is data, not architecture. Every
component downstream of the parsers carries the sensor identities alongside the
values, and no component embeds an assumption about how many channels there are.
This single decision is what allows one checkpoint to serve any asset, and it
propagates all the way into the exported graph, whose sensor dimension is
declared dynamic.

A second principle is that the training pipeline and the serving pipeline must
share code wherever they compute the same quantity. The normalisation and
feature modules under `unify/common/` are pure NumPy and are imported by both
the trainer and the edge service, so that a change to how a window is normalised
cannot silently apply to one and not the other. Section 5.6 documents what
happens when the two paths diverge even slightly.

*[Figure 3.1: Proposed System Architecture Block Diagram]*

*[Figure 3.2: End-to-End Data Flow Flowchart]*

### 3.2 Dataset Model – C-MAPSS Turbofan Degradation

The C-MAPSS dataset is the primary experimental vehicle for this project. It was
generated by NASA's Commercial Modular Aero-Propulsion System Simulation, which
models a large turbofan engine and injects a progressive degradation into one or
more of its rotating components. Each engine unit is recorded as a multivariate
time series over flight cycles, with three operational setting variables and
twenty-one sensor measurements per cycle.

The four subsets differ in the number of concurrent operating conditions and
fault modes, and therefore in difficulty. FD001 and FD003 run under a single
operating condition, while FD002 and FD004 cycle through six, which makes the
sensor readings of a healthy engine vary as much with what the engine is doing
as with how damaged it is.

| Subset | Operating Conditions | Fault Modes | Train Units | Test Units |
|---|---|---|---|---|
| FD001 | 1 | 1 (HPC degradation) | 100 | 100 |
| FD002 | 6 | 1 (HPC degradation) | 260 | 259 |
| FD003 | 1 | 2 (HPC and fan) | 100 | 100 |
| FD004 | 6 | 2 (HPC and fan) | 249 | 248 |

*Table 3.1: C-MAPSS Subset Characteristics*

All four subsets are used jointly. After parsing, the cached corpus contains
1,416 units and 265,256 timesteps. Six of the twenty-one sensors are
near-constant across the whole corpus and carry no degradation information; they
are dropped in the parser, leaving fifteen informative channels with sensor
identities 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20 and 21. These
identities, not positional indices, are what the model is given, so that the same
channel means the same thing to the model regardless of which subset or which
asset it came from.

The published `test_FDxxx.txt` files are separated by a dedicated hold-out
function keyed on the unit identifier and are never seen during training or
model selection. This distinction matters: the validation split drives early
stopping and checkpoint selection, so quoting it as the headline result would be
selection bias, and only the official test split is comparable to published
literature.

### 3.3 Preprocessing and Per-Condition Normalisation

Sensor channels differ by orders of magnitude in scale, so each is standardised
to zero mean and unit variance before entering the model. The statistics are
fitted on the training units only. Fitting them on the full corpus would leak
information about the held-out engines into the model's input distribution and
would inflate the reported result.

For C-MAPSS the statistics are fitted **per operating condition** rather than
globally. In FD002 and FD004 the operating condition changes the sensor readings
far more than degradation does, so a single global mean and standard deviation
per channel would place the degradation signal inside the noise of the condition
variation. Conditioning the statistics removes the operating-condition component
and leaves the degradation trend visible. The magnitude of this effect is large:
for the fifteen C-MAPSS channels, the global and per-condition tables differ by
up to 390 units in the mean and by a factor of 1,105 in the standard deviation.
Section 5.6 describes what happened when the serving path used the wrong one.

The normalised series is then cut into overlapping windows of L = 30 consecutive
timesteps with a stride of one. A window is the model's unit of input; it is
presented as an array of shape (S, L) together with the vector of S sensor
identities, a boolean mask of which channels are valid, and an integer domain
identifier naming the source dataset.

### 3.4 Sensor-Agnostic Model Architecture

The model receives a window as a set of S independent channel sequences rather
than as a single S-dimensional multivariate sequence. Each channel's length-L
sequence is passed through a shared one-dimensional convolutional encoder,
producing one d-dimensional vector per channel. A learned embedding of the
channel's sensor identity and a learned embedding of the domain identifier are
added to that vector, so the representation carries not only what the channel
did but which channel it was and which asset family it came from.

The resulting set of S tokens is pooled by a single-query multi-head attention
layer into one fixed-size vector, which a small feed-forward head maps to a
scalar in the interval [0, 1]. Because pooling is by attention over a set, the
number of tokens is free: S may be two or twenty-one and the downstream shapes
are unchanged. Channels marked invalid by the mask are excluded from the
attention softmax, so a failed sensor removes its token rather than contributing
a fabricated value.

Normalisation inside the encoder is GroupNorm rather than BatchNorm. Because the
sensor axis is folded into the batch dimension for the shared encoder, BatchNorm
statistics would be contaminated by zero padding in batches that mix assets with
different sensor counts; GroupNorm is batch-independent and does not have this
failure mode.

| Parameter | Value |
|---|---|
| Total trainable parameters | 44,417 |
| Model dimension d | 64 |
| Attention heads | 4 |
| Attention queries | 1 |
| Window length L | 30 timesteps |
| Sensor vocabulary size | 64 |
| Domain embedding size | 8 |
| Encoder normalisation | GroupNorm |
| Dropout / attention dropout | 0.2 / 0.1 |
| Output | scalar in [0, 1], clamped at evaluation |

*Table 3.2: Model Architecture Parameters*

### 3.5 Piecewise RUL Labelling and Loss Function

The target is the normalised remaining fraction of life, not the raw remaining
time. Each dataset carries a life scale which converts the model's [0, 1] output
back into that dataset's own unit — 125 cycles for C-MAPSS. This is what allows
datasets recorded in cycles, seconds and minutes to train jointly against one
output head: without it, the dataset with the largest raw numbers would dominate
the loss and the others would be fitted only incidentally.

The label itself is piecewise linear with a cap. While an engine is healthy its
sensors are essentially flat, so a target that decreases linearly from a large
initial value would demand that the model distinguish states that are not
distinguishable from the data. Capping the target at 125 cycles during the
healthy phase and letting it descend linearly thereafter concentrates the
model's capacity on the degradation phase, which is the phase a maintenance
decision actually depends on.

*[Figure 3.3: Piecewise Linear RUL Labelling with Cap at 125 Cycles]*

Training minimises a SmoothL1 loss with a transition point of 0.1 in normalised
units, which behaves quadratically for small errors and linearly for large ones
and is therefore less sensitive to the outlier windows that occur near a
transition. Batches deliberately mix datasets on every optimisation step; training
one dataset to convergence before moving to the next causes catastrophic
forgetting, and only mixed batches force the single shared representation on
which the universality claim depends. Inverse-frequency sampling balances the
datasets so that the largest does not dominate by sheer window count.

| Hyperparameter | Value |
|---|---|
| Optimiser | AdamW |
| Learning rate | 1.0 × 10⁻³ with 5-epoch warmup, cosine to 1.0 × 10⁻⁵ |
| Weight decay | 1.0 × 10⁻⁴ |
| Batch size | 256 |
| Gradient clipping | 1.0 |
| Loss | SmoothL1, β = 0.1 |
| Maximum epochs | 100 |
| Early-stop patience | 15 epochs |
| Validation fraction | 0.2 of units, never of windows |
| Random seed | 42 |

*Table 3.3: Training Hyperparameters*

### 3.6 Alarm Logic and Hysteresis

A raw model output is not an alarm. Window-to-window predictions fluctuate, and
a bare threshold on a fluctuating signal produces alarm chatter that maintenance
staff learn to ignore, which destroys the value of the system more thoroughly
than a missed detection would.

The service therefore smooths the predicted health with an exponentially
weighted moving average using α = 0.3, and raises an alarm only after the
smoothed health has remained below a threshold of 0.30 for five consecutive
predictions. The two mechanisms address different problems: the EWMA suppresses
single-sample noise, and the consecutive-count requirement suppresses sustained
excursions that are still too brief to be a real trend. Section 5.6 reports the
resulting lead time and false-alarm count on a real engine.

---

## CHAPTER 4  IMPLEMENTATION

### 4.1 Training Pipeline Implementation

The pipeline is implemented in Python with PyTorch and is organised as
importable modules rather than scripts, so that the same code paths are
exercised by the tests, by the trainer and by the evaluation tooling. Dataset
parsing produces cached `.npz` units carrying signals, sensor identities, RUL,
operating condition and metadata; caching is content-addressed by dataset name
so that a rebuild is explicit rather than accidental.

Before any real training run the pipeline is required to pass a capacity check:
the model is asked to memorise eight samples with dropout disabled, and the
maximum absolute error must fall to approximately zero. If a model cannot
overfit eight samples the fault is in the model or the data path, and no number
of epochs will repair it. On the final configuration this check reached a maximum
absolute error of 0.00030 in normalised life units.

Training reports metrics per dataset and in physical units at every epoch, never
only as a mean. A mean across datasets can conceal a total collapse on one of
them, and a normalised RMSE is not comparable to any published number.
Checkpoint selection is on validation RMSE, and the selected checkpoint — not
the final epoch's weights — is the one evaluated on the official test split and
the one exported.

### 4.2 ONNX Export Implementation

Export uses the legacy TorchScript tracing path, explicitly requested with
`dynamo=False`. On the installed PyTorch version the newer torch.export-based
exporter ignores the requested opset version, drops the dynamic axes from the
output shapes, and emits a graph that the quantisation tool rejects; the tracing
path handles all three correctly. The export targets opset 17 and declares the
batch and sensor axes as dynamic.

Export is followed immediately by an automated verification stage, because an
export that silently changes behaviour is worse than an export that fails. The
verifier runs the ONNX graph and the PyTorch model on identical inputs across a
range of sensor counts and compares outputs elementwise.

| Sensor count S | ONNX output | Torch output | Max abs difference | Status |
|---|---|---|---|---|
| 2 | 0.592407 | 0.592407 | 1.19 × 10⁻⁷ | ok |
| 10 | 0.721884 | 0.721884 | 1.19 × 10⁻⁷ | ok |
| 14 | 0.637798 | 0.637798 | 5.96 × 10⁻⁸ | ok |
| 21 | 0.430928 | 0.430928 | 2.98 × 10⁻⁸ | ok |
| batch = 8 | shape (8, 1) | shape (8, 1) | — | ok |
| one channel masked | finite, delta 0.0250 | — | — | ok |

*Table 4.1: ONNX Export Verification Across Sensor Counts*

The dynamic sensor axis is therefore not a design intention but a verified
property of the shipped artifact: the same graph accepts two channels and
twenty-one channels, and masking a channel changes the output by a finite,
sensible amount rather than producing a NaN.

### 4.3 INT8 Quantisation

Dynamic INT8 quantisation is applied to the verified FP32 graph, and the export
tool refuses to treat the size reduction as a result on its own — accuracy must
be re-measured on the same official test windows afterwards. That measurement is
implemented as a separate evaluation stage which runs the PyTorch checkpoint, the
FP32 ONNX graph and the INT8 ONNX graph over identical inputs and reports all
three side by side.

| Artifact | Size | Reduction |
|---|---|---|
| PyTorch checkpoint (`best.pt`) | 189 KB | — |
| ONNX FP32 (`unify_rul.onnx`) | 190.4 KB | 1.0× |
| ONNX INT8 (`unify_rul_int8.onnx`) | 79.3 KB | 2.4× |

*Table 4.2: Deployment Footprint After Quantisation*

*[Figure 4.1: Deployment Footprint – Checkpoint, FP32 ONNX, INT8 ONNX]*

The accuracy consequence of this reduction is reported in Section 5.5, together
with the runtime limitation that prevents the quantised artifact from being used
on the target board at present.

### 4.4 Edge Inference Service Implementation

The edge service is a FastAPI application backed by ONNX Runtime with the CPU
execution provider. It exposes an ingest endpoint that accepts a feature packet
from a sensor node, a health endpoint that reports service state and the active
model hash, and a static dashboard.

Incoming packets are appended to a fixed-length deque of L = 30 timesteps. Until
the deque is full the service replies `warming_up` rather than predicting from a
partial window, and it tracks sequence numbers so that dropped packets are
counted as gaps instead of being silently concatenated across a discontinuity.
Once full, the window is normalised with the statistics staged alongside the
model, run through the graph, smoothed by the EWMA, evaluated against the alarm
hysteresis, and written to SQLite with its latency.

The mapping from the physical rig's channels to the model's trained sensor
identities lives in a separate live-map configuration rather than in code, so a
new asset is a configuration change. The map names the reference dataset that
supplies the normalisation statistics and life scale, the domain identifier, the
ordered channel list, and — following the finding in Section 5.6 — the operating
condition under which those statistics should be looked up.

| Endpoint / artifact | Purpose |
|---|---|
| `POST /ingest` | accept one feature packet; returns status, RUL, health, alarm |
| `GET /health` | uptime, window fill, gaps, rejected packets, model hash, active condition |
| `GET /` | live dashboard |
| `edge/artifacts/` | ONNX graph, normalisation statistics, sensor vocabulary, manifest, model hash |
| `edge/unify.db` | SQLite log of every prediction with latency and attention weights |

*Table 4.3: Edge Service Interface Summary*

Every artifact carries a SHA-256 hash, and the service reports it on the health
endpoint. Without this the deployed build cannot be identified, and a benchmark
cannot be attributed to a model — a failure mode encountered during this project,
where an earlier benchmark file on the board recorded a model hash matching no
artifact present on it.

### 4.5 Sensor Node and Replay Implementation

Two node implementations post packets to the ingest endpoint. The first is
firmware for an ESP32 with an MPU-6050 inertial sensor, which computes RMS,
kurtosis and crest factor per axis and posts them at 1 Hz. The second is a
software node used for all validation in this report, which can either generate
synthetic degradation or replay a cached real unit cycle by cycle.

The replay mode is the important one: it is the only mode whose feature
distribution matches the training distribution, and it is therefore the only mode
whose predictions can be checked against a known ground truth. The node queries
the service's health endpoint for the expected channel list rather than
hard-coding it, so the node and the service cannot disagree about channel order.

It is recorded here as a limitation that the physical rig cannot validate bearing
RUL. The MPU-6050 tops out at a 1 kHz output data rate against the bearing
datasets' 25.6 kHz, so the rig demonstrates the pipeline and low-frequency fault
detection such as imbalance and looseness, and a physical RUL figure from it
would be an unvalidated transfer rather than a measurement.

---

## CHAPTER 5  EXPERIMENTS AND RESULTS

### 5.1 Experimental Setup and Environment

All training was performed on a Windows 11 workstation with a CUDA-capable GPU,
and all edge measurements on a Raspberry Pi 5 with CPU-only inference. The two
environments are reported separately throughout, because a latency measured on
the workstation would say nothing about the deployed system.

| Parameter | Value |
|---|---|
| Training framework | PyTorch 2.13.0+cu130 |
| Export and evaluation runtime | ONNX Runtime 1.28.0 (workstation) |
| Edge board | Raspberry Pi 5 Model B Rev 1.1, 4 cores, 8 GB |
| Edge OS kernel | Linux 6.12.93-rpt-rpi-2712, aarch64 |
| Edge runtime | ONNX Runtime 1.20.1, CPUExecutionProvider |
| Dataset | NASA C-MAPSS, subsets FD001–FD004 |
| Corpus after parsing | 1,416 units, 265,256 timesteps, 15 channels |
| Training windows | 112,130 |
| Validation windows | 27,668 |
| Official test engines | 690 |
| Window length | 30 cycles |
| Life scale | 125 cycles |
| Random seed | 42 |
| Automated tests | 61, all passing |

*Table 5.1: Experimental Environment Parameters*

The dataset cache was rebuilt from the raw files for these experiments rather
than reused, so that the reported results verify the current code rather than the
history of previous runs.

### 5.2 Training Convergence

Training ran for 28 epochs and stopped early after fifteen consecutive epochs
without improvement. Validation RMSE fell from above 20 cycles in the first ten
epochs to 15.02 cycles at epoch 12, which was selected as the shipping
checkpoint. Training loss continued to decrease after that point while validation
RMSE oscillated between 15 and 18 cycles, which is the signature of a model that
has stopped generalising further and is the behaviour the early-stopping criterion
exists to catch.

*[Figure 5.1: Training Convergence – Validation RMSE and Training Loss]*

The gap between the training loss curve and the validation RMSE curve remains
modest throughout, which is consistent with a 44,417-parameter model on 112,130
training windows: the model is small enough relative to the data that overfitting
is not the binding constraint.

### 5.3 Prediction Accuracy Analysis

The selected checkpoint was evaluated on the official C-MAPSS test split — the
published `test_FDxxx.txt` engines, one prediction per engine from its final
window, never used in training or model selection.

*[Figure 5.2: Predicted vs True RUL on the Official Test Split]*

Across 690 engines the model attains an RMSE of 14.895 cycles, an MAE of 10.188
cycles and a NASA score of 3,831. Five hundred of the 690 engines, or 72.5 %,
are predicted within ±13 cycles of their true remaining life. The scatter is
tight along the diagonal in the low-RUL region, which is the region a maintenance
decision depends on, and widens in the high-RUL region where the sensors carry
little degradation information.

The vertical band of points at a true RUL of 125 cycles is an artefact of the
piecewise labelling rather than a model failure: every engine still in its
healthy phase carries the same capped target, so the model cannot be graded on
those points and should not be.

*[Figure 5.3: Error Distribution on the Official Test Split]*

Separating the error by sign is more informative than its magnitude. An early
prediction wastes a service slot; a late prediction lets the machine fail. The
distribution is close to symmetric about zero, which is honest but not the ideal
operating point for a maintenance system, where a conservative bias is worth
paying for. The pinball loss with an alarm on the tenth percentile is implemented
in the codebase as the mechanism for introducing that bias and is identified in
the future scope as the next configuration to evaluate.

### 5.4 Comparative Results: Proposed Model vs Classical Baseline

A RandomForest regressor over 105 engineered features — statistical summaries
and trend features computed per window per channel — was trained on the identical
split as a classical reference.

| Metric | UNIFY-RUL (44,417 params) | RandomForest (105 features) | Better |
|---|---|---|---|
| RMSE (cycles) | 14.895 | 14.405 | RandomForest |
| MAE (cycles) | 10.188 | 10.497 | UNIFY-RUL |
| NASA score | 3,831 | 2,558 | RandomForest |
| Validation RMSE (cycles) | 15.024 | 14.843 | RandomForest |
| Deployed artifact size | 79.3 KB | tens of MB | UNIFY-RUL |
| Accepts variable sensor count | yes | no | UNIFY-RUL |
| Inference on Pi 5 (p95) | 0.675 ms | not deployed | UNIFY-RUL |

*Table 5.2: Accuracy Comparison – Proposed Model vs RandomForest*

*[Figure 5.4: Proposed Model vs RandomForest Baseline]*

The RandomForest is more accurate on RMSE and on the NASA score, and this report
states that plainly rather than omitting the baseline. On a single dataset with
good hand-built features and no deployment constraint, a 44,417-parameter
attention model buys nothing in accuracy, and any claim to the contrary would not
survive the table above.

What the comparison does establish is that the accuracy cost of the proposed
architecture is small — under half a cycle of RMSE — and that it is paid for
capabilities the baseline does not have at any price. The RandomForest's feature
extractor is defined against fifteen specific channels in a specific order; it
cannot accept eleven channels, cannot tolerate one failing at runtime, and a
200-tree ensemble over 105 features is not a 79 KB artifact that predicts in
0.675 ms on a CPU-only board. The defensible claim for this architecture is
deployability and sensor-layout independence at near-parity accuracy, and that is
the claim this report makes.

### 5.5 Edge Deployment Performance

The complete service, artifacts and configuration were deployed to the Raspberry
Pi 5. Latency was benchmarked over 1,000 inferences per configuration after 50
warm-up iterations, with the board otherwise idle — an unrelated service that had
been saturating all four cores was stopped first, since any measurement taken
beside it would have described that service rather than this one.

| Threads | p50 (ms) | p95 (ms) | p99 (ms) | Max (ms) |
|---|---|---|---|---|
| 1 | 0.886 | 0.907 | 0.926 | 1.056 |
| 2 | 0.723 | 0.748 | 0.758 | 0.854 |
| 3 | 0.662 | 0.686 | 0.693 | 0.795 |
| 4 | **0.654** | **0.675** | 0.684 | 0.846 |

*Table 5.3: Inference Latency Across Thread Counts*

*[Figure 5.5: Inference Latency Sweep on Raspberry Pi 5]*

Percentiles are reported rather than means, because a real-time system is
characterised by its tail and not by its average. Scaling from one to four
threads improves the median by only 26 %, and from three to four threads by under
1.5 %: for a model this small, thread synchronisation costs nearly as much as the
parallelism returns. SoC temperature rose from 43.9 °C to 48.8 °C across the full
sweep of 4,000 inferences with the throttling register reading `0x0` throughout,
so none of these numbers were taken under thermal limitation.

*[Figure 5.6: Inference Cost Against the Real-Time Budget]*

Against a one-second sensing period, a 0.675 ms p95 inference consumes 0.068 % of
the available budget, leaving approximately 1,481× of headroom. The operative
conclusion is that the model is not the bottleneck in this system — the sensing
window is — which is the correct result for an edge design and is what makes a
multi-node deployment on a single board plausible.

The accuracy of each conversion step was re-measured on the identical official
test windows rather than assumed.

| Artifact | RMSE | MAE | NASA score | Max abs deviation vs PyTorch |
|---|---|---|---|---|
| PyTorch FP32 | 14.895 | 10.188 | 3,831.3 | — |
| ONNX FP32 | 14.895 | 10.188 | 3,831.3 | 0.000 cycles |
| ONNX INT8 | 14.938 | 10.311 | 3,788.9 | 11.368 cycles |

*Table 5.4: Post-Export and Post-Quantisation Accuracy*

The FP32 export is numerically identical to the PyTorch model on this data, so
the export step costs nothing. INT8 quantisation costs 0.044 cycles of RMSE, or
0.29 %, for a 2.4× size reduction — an excellent trade in aggregate. The final
column is a necessary caveat: individual engines move by as much as 11.4 cycles
under INT8 even though the aggregate barely changes, so a per-asset decision
threshold calibrated on FP32 outputs must be re-derived if INT8 is adopted.

One deployment limitation must be recorded. The INT8 graph runs correctly on the
workstation under ONNX Runtime 1.28.0, but on the board's ONNX Runtime 1.20.1 for
aarch64 it fails at session creation with `NOT_IMPLEMENTED: Could not find an
implementation for ConvInteger(10)`. The 2.4× saving is therefore not currently
realisable on this target, and the FP32 graph is what ships. Since the FP32
artifact is 190.4 KB on a board with 8 GB of memory, the practical cost of this
limitation is nil, but it is stated rather than glossed because the quantisation
figure would otherwise read as a deployed result.

### 5.6 Live Run-to-Failure Validation

The final experiment streamed a real C-MAPSS FD001 engine through the deployed
service one cycle at a time, exactly as a physical sensor node would post it. The
unit runs 199 cycles from a true RUL of 198 down to 0, producing 170 predictions
after the 30-cycle warm-up.

*[Figure 5.7: Live Run-to-Failure Trace on the Raspberry Pi 5]*

The predicted RUL remains near the 125-cycle cap while the engine is healthy,
then tracks the true remaining life down the degradation slope, converging onto
the ground-truth line and reaching exactly 0.0 at the true end of life. Over the
region where the true RUL lies within the cap, the trace RMSE is 11.35 cycles.
The alarm first fires at cycle 166 — 32 cycles of warning before failure — and
remains asserted for the final 33 cycles. There were no false alarms during the
137 healthy cycles preceding it. All 170 packets were accepted with zero gaps
and zero rejections.

*[Figure 5.8: End-to-End Service Latency Distribution]*

End-to-end latency through the complete service — deserialisation, windowing,
normalisation, inference, alarm logic and SQLite persistence — is 1.191 ms at the
median, 1.220 ms at the 95th percentile and 1.418 ms at maximum. The service
layer therefore adds roughly 0.5 ms on top of the bare inference cost, and the
whole path remains three orders of magnitude inside the sensing period.

*[Figure 5.9: Edge Dashboard Served from the Raspberry Pi 5]*

The dashboard in Figure 5.9 is the operator-facing end of the same service,
served by the Pi itself and read over the local network from a laptop browser.
It is shown here at the close of the replay: the health index has fallen to 1 %,
the predicted remaining useful life reads 0.0 cycles, the maintenance banner is
asserted, and the header reports the sensor node as offline because the replay
has finished posting. The measured inference latency of 1.29 ms and the model
hash `onnx:a80c367e596a6517` are the figures of Sections 5.5 and 4.2 as the
running service reports them, not values transcribed into the page. The
attention panel shows which of the fifteen channels the model weighted for the
prediction on screen, sensors 11, 4 and 14 carrying most of it.

**A serving-path defect exposed by this experiment.** The first execution of this
test ended with the deployed service reporting an RUL of 65.1 cycles for an
engine that had already failed, with the alarm never firing because the smoothed
health bottomed out at 0.53 against a threshold of 0.30. The identical checkpoint,
evaluated offline on the identical engine, returned 0.00. Every offline metric in
Sections 5.2 to 5.4 had passed while the deployed system was producing a
confidently wrong answer.

The cause was located in the serving path's normalisation lookup. The predictor
requested normalisation statistics without supplying an operating condition, and
the lookup silently falls back to the global table when no condition is given,
whereas training had used the per-condition table described in Section 3.3.
Feeding the offline model the global statistics reproduced the deployed answer to
the decimal — 65.10 against the service's 65.10 — confirming the diagnosis
rather than merely making it plausible. The two tables differ by up to 390 units
in the mean and by a factor of 1,105 in the standard deviation, so the model was
being shown inputs far outside its trained distribution and was, correctly for a
regressor, returning a confident number anyway.

The fix is a single guard placed where every caller passes through: the live map
may declare the operating condition, and a reference dataset whose statistics are
per-condition now **refuses to load** without one, naming the valid identifiers
in the error. A silent fallback was the defect, so a default would have been the
wrong repair; the correct repair is a refusal. Four regression tests pin the
behaviour, including one that verifies the two statistics tables genuinely differ
so that the others cannot pass for the wrong reason. Redeployed, the same replay
ends at an RUL of 0.0 with the alarm asserted, as reported above.

This result is presented at the same level as the accuracy results because it is
the clearest evidence in this project for the design position taken in
Section 2.5: a prognostic system can be entirely correct as a model and entirely
wrong as a service, and only end-to-end validation on the target hardware can
tell the two apart.

| Metric | Value | Notes |
|---|---|---|
| Official test RMSE | 14.895 cycles | 690 held-out engines |
| Official test MAE | 10.188 cycles | — |
| NASA score | 3,831 | asymmetric, penalises late predictions |
| Engines within ±13 cycles | 72.5 % | 500 of 690 |
| Model size | 44,417 parameters | 190.4 KB FP32, 79.3 KB INT8 |
| ONNX export deviation | 0.000 cycles | FP32, identical to PyTorch |
| Quantisation cost | +0.29 % RMSE | 2.4× size reduction |
| Verified sensor-count range | S = 2 to 21 | one unchanged graph |
| Pi 5 inference latency | 0.654 ms p50, 0.675 ms p95 | 4 threads, CPU only |
| Pi 5 service latency | 1.191 ms p50, 1.220 ms p95 | full REST path |
| Real-time budget consumed | 0.068 % | 1,481× headroom at 1 Hz |
| Live trace RMSE | 11.35 cycles | within the RUL cap |
| Alarm lead time | 32 cycles | before true failure |
| False alarms | 0 | across 137 healthy cycles |
| Automated tests | 61 passing | includes 4 regression tests for §5.6 |

*Table 5.5: Performance Summary Table*

---

## CHAPTER 6  CONCLUSION AND FUTURE SCOPE

### 6.1 Conclusion

This project has designed, implemented, deployed and validated UNIFY-RUL, a
sensor-agnostic remaining-useful-life prediction system that runs on low-cost
edge hardware. The system addresses two limitations that keep published RUL
models out of practical maintenance settings — architectural dependence on a
fixed sensor layout, and inference weight unsuited to on-site hardware — by
treating the sensor axis as data rather than as architecture, which makes the
resulting model both layout-independent and small.

The key contributions of this work are as follows:

(i) A remaining-useful-life architecture in which each sensor channel is a token
carrying a learned identity embedding and channels are pooled by attention, so
that one set of 44,417 trained weights accepts an arbitrary sensor count in an
arbitrary order. This is not merely a design intention: the exported graph was
verified to produce correct outputs for sensor counts from 2 to 21 without
modification, and to handle a masked channel with a finite, sensible change in
output.

(ii) A normalised-life formulation in which the model predicts a fraction of
life in [0, 1] and denormalises through a per-dataset life scale, allowing
datasets recorded in different units of time to share one output head without
the largest-numbered dataset dominating the loss.

(iii) Accuracy on the published NASA C-MAPSS test split of 14.895 cycles RMSE,
10.188 cycles MAE and a NASA score of 3,831 across 690 held-out engines, with
72.5 % of engines predicted within ±13 cycles — measured on the official test
files rather than on a private split, and therefore comparable to published
literature.

(iv) A verified deployment path in which accuracy is re-measured after every
conversion: ONNX export reproduces the PyTorch model to 0.000 cycles on the test
data, and INT8 quantisation reduces the artifact from 190.4 KB to 79.3 KB at a
cost of 0.29 % RMSE. The runtime limitation that prevents the quantised artifact
from executing on the target board's ONNX Runtime version is documented rather
than omitted.

(v) Measured edge performance on a Raspberry Pi 5 with CPU-only inference of
0.654 ms median and 0.675 ms 95th-percentile per prediction, and 1.191 ms median
through the complete REST service, consuming 0.068 % of a one-second sensing
budget under verified non-throttled thermal conditions.

(vi) End-to-end validation on real degradation data, in which a C-MAPSS engine
replayed cycle by cycle through the deployed service was tracked to an RUL of
0.0 at its true end of life, with an alarm raised 32 cycles before failure and no
false alarms across the preceding 137 healthy cycles.

(vii) The identification, root-cause diagnosis, repair and regression testing of
a serving-path normalisation defect that every offline metric had passed and that
produced a confidently incorrect prediction on hardware — reported here as a
result because it demonstrates concretely why edge prognostic systems require
end-to-end validation on the target device rather than offline evaluation alone.

An honest limitation is stated alongside these contributions. A RandomForest
baseline over engineered features achieves a marginally better RMSE and NASA
score on this single dataset. The case for the proposed architecture is therefore
not accuracy superiority but near-parity accuracy combined with capabilities the
baseline does not possess: a variable sensor axis, tolerance of channel failure
at runtime, a 79 KB deployable artifact, and sub-millisecond CPU inference on a
board costing under a hundred dollars.

The system is well suited to condition monitoring of rotating machinery in
manufacturing, distributed monitoring of wind and gas turbines, fleet vehicle
and rail powertrain monitoring, and any setting where assets are heterogeneous
enough that maintaining one model per machine type is impractical. The design
philosophy — treat the sensor layout as configuration, size for the target board
from the first line of code, and trust no accuracy figure that has not been
re-measured on the artifact that ships — is the transferable outcome of this
work.

### 6.2 Future Scope

The work presented in this project opens several avenues for future research and
development, particularly in demonstrating the generality the architecture was
designed for and in hardening the system for sustained field deployment.

**(i) Joint training across FEMTO-ST and XJTU-SY.**
The architecture was designed for cross-dataset universality, but only C-MAPSS
was available for these experiments, so the central claim of the project title
remains demonstrated in mechanism rather than in results. Acquiring the FEMTO-ST
and XJTU-SY bearing datasets and training jointly across all three would test
whether a single checkpoint can serve turbofan engines and rolling bearings at
once, and would exercise the normalised-life formulation across genuinely
incompatible units of time. This is the single highest-value extension.

**(ii) Quantile regression for conservative alarming.**
The current model produces a point estimate and its error distribution is close
to symmetric, whereas a maintenance system should be biased toward early
warnings. The pinball loss with quantiles at 0.1, 0.5 and 0.9 is already
implemented; training with it and raising alarms on the tenth-percentile
prediction would give both a conservative bias and a prediction interval,
converting each output from a number into a number with a stated confidence.

**(iii) A physical high-bandwidth instrumented rig.**
The present ESP32 and MPU-6050 node is limited to a 1 kHz output data rate
against the 25.6 kHz of the bearing benchmarks, so it can demonstrate the
pipeline and low-frequency faults but cannot validate bearing RUL. Replacing it
with a piezoelectric accelerometer and a suitable high-rate acquisition front
end would allow the trained model to be validated against a physically
instrumented machine rather than against replayed data.

**(iv) Resolving INT8 execution on the target runtime.**
The quantised artifact cannot execute on the board's current ONNX Runtime
because of a missing `ConvInteger` kernel for aarch64. Upgrading the runtime,
re-quantising with a scheme that avoids the unsupported operator, or evaluating
alternative edge runtimes would make the 2.4× size reduction realisable in
deployment, which matters as soon as the target moves from a Raspberry Pi to a
microcontroller-class device.

**(v) Multi-node deployment and horizontal scaling.**
The measured 1,481× real-time headroom implies that a single board could serve a
large number of sensor nodes concurrently, but this has not been measured. A
scaling study establishing how many nodes one board sustains before tail latency
degrades, together with per-node model and configuration isolation, would turn
the headroom figure into a deployment specification.

**(vi) On-device continual learning and drift detection.**
Degradation behaviour changes as an asset ages and as operating conditions
shift, so a model frozen at deployment slowly becomes wrong. Monitoring the
distribution of incoming windows against the training distribution would detect
this drift, and lightweight on-device adaptation of the output head — or
federated aggregation of updates across sites without moving raw sensor data —
would let a fleet improve its models while keeping proprietary operating data
on-site.

**(vii) Long-duration field trial with maintenance-outcome tracking.**
Every result in this report is derived from benchmark data or from replay of it.
A field trial on a real machine over months, logging every alarm and every
subsequent maintenance finding, is what would convert the technical metrics of
Chapter 5 into the operational metrics that justify a predictive maintenance
programme: false-alarm rate, missed-detection rate, achieved lead time, and
avoided downtime.

---

## CHAPTER 7  APPENDICES

### APPENDIX 1 — PYTHON CODE: SENSOR-AGNOSTIC PREDICTION AND EDGE INFERENCE

The following is a representative excerpt of the implementation. The complete
codebase implements dataset parsing, per-condition normalisation, windowing and
piecewise RUL labelling, the attention model, joint training, ONNX export and
verification, quantisation, and the edge inference service.

```python
# Attention pooling over the sensor axis: the number of channels is data,
# not architecture. S may be 2 or 21 and the shapes below do not change.
def forward(self, x, mask, sensor_ids, domain_id):
    B, S, L = x.shape
    h = self.encoder(x.reshape(B * S, 1, L)).reshape(B, S, -1)
    h = h + self.sensor_emb(sensor_ids) + self.domain_emb(domain_id)[:, None]
    q = self.query.expand(B, -1, -1)
    pooled, attn = self.attn(q, h, h, key_padding_mask=~mask)
    return self.head(pooled.squeeze(1)), attn


# Piecewise RUL label, capped during the healthy phase.
def rul_label(cycle, life_total, cap=125.0):
    return min(life_total - cycle, cap) / cap


# Edge inference. The condition is mandatory for a per-condition dataset:
# without it the global statistics are used and the prediction is wrong by
# tens of cycles while still looking plausible.
if stats["datasets"][ref_dataset].get("per_condition") and condition is None:
    raise ArtifactsMissing(
        f"{ref_dataset} was normalised per operating condition during "
        f"training, so this live map must set \"condition\"")

mean, std = norm.vector_for(ref_dataset, sensor_ids, condition)


def predict(self, window):
    x = ((window - self.mean) / self.std).T[None].astype(np.float32)
    feeds = {"x": x,
             "mask": np.ones((1, x.shape[1]), dtype=bool),
             "sensor_ids": self.sensor_ids[None],
             "domain_id": np.array([self.domain_id], dtype=np.int64)}
    rul, attn = self.sess.run(["rul", "attn"], feeds)
    health = float(np.clip(rul[0], 0.0, 1.0)[0])
    return {"health": health, "rul": health * self.life_scale}
```

### APPENDIX 2 — CONSOLE OUTPUT: LIVE RUN-TO-FAILURE DEMONSTRATION

The following is the recorded console output from the deployed Raspberry Pi 5
service while a real C-MAPSS FD001 engine was replayed from health to failure.

```
================================================================
UNIFY-RUL edge service - Raspberry Pi 5, CPU only
model onnx:a80c367e596a6517   ref=cmapss  scale=125  cond=0
15 channels: cmapss/s2 s3 s4 s6 s7 s8 s9 s11 s12 s13 s14 s15 s17 s20 s21
================================================================
 cycle | predicted RUL | health | alarm
-------+---------------+--------+-------
    29 |        116.86 |  0.935 | false     <- warm-up complete
    75 |        121.40 |  0.971 | false
   125 |         73.55 |  0.588 | false     <- degradation begins
   150 |         54.12 |  0.433 | false
   166 |         26.41 |  0.297 | TRUE      <- first alarm, 32 cycles early
   180 |         14.86 |  0.161 | true
   194 |          2.82 |  0.043 | true
   195 |          1.31 |  0.033 | true
   196 |          1.01 |  0.026 | true
   197 |          0.00 |  0.018 | true
   198 |          0.00 |  0.013 | true      <- true end of life
================================================================
170 predictions, 0 gaps, 0 rejected, 0 false alarms
latency  p50 1.191 ms   p95 1.220 ms   p99 1.259 ms   max 1.418 ms
replay finished (unit reached end of life)
================================================================
```

---

## CHAPTER 8  REFERENCES

1. Babu G.S., Zhao P. and Li X.L. (2016) 'Deep Convolutional Neural Network
   Based Regression Approach for Estimation of Remaining Useful Life of
   Machinery', Proceedings of the 21st International Conference on Database
   Systems for Advanced Applications, pp. 214–228.

2. Banbury C., Reddi V.J., Lam M., Fu W., Fazel A., Holleman J., Huang X.,
   Hurtado R., Kanter D., Lokhmotov A., Patterson D., Pau D., Seo J., Sieracki
   J., Thakker U., Verhelst M. and Yadav P. (2021) 'Benchmarking TinyML Systems:
   Challenges and Direction', Proceedings of Machine Learning and Systems,
   Vol. 3, pp. 800–811.

3. Chen Z., Wu M., Zhao R., Guretno F., Yan R. and Li X. (2021) 'Machine
   Remaining Useful Life Prediction via an Attention-Based Deep Learning
   Approach', IEEE Transactions on Industrial Electronics, Vol. 68, No. 3,
   pp. 2521–2531.

4. Jacob B., Kligys S., Chen B., Zhu M., Tang M., Howard A., Adam H. and
   Kalenichenko D. (2018) 'Quantization and Training of Neural Networks for
   Efficient Integer-Arithmetic-Only Inference', Proceedings of the IEEE
   Conference on Computer Vision and Pattern Recognition, pp. 2704–2713.

5. Jardine A.K.S., Lin D. and Banjevic D. (2006) 'A Review on Machinery
   Diagnostics and Prognostics Implementing Condition-Based Maintenance',
   Mechanical Systems and Signal Processing, Vol. 20, No. 7, pp. 1483–1510.

6. Lee J., Wu F., Zhao W., Ghaffari M., Liao L. and Siegel D. (2014)
   'Prognostics and Health Management Design for Rotary Machinery Systems —
   Reviews, Methodology and Applications', Mechanical Systems and Signal
   Processing, Vol. 42, No. 1–2, pp. 314–334.

7. Lee J., Bagheri B. and Kao H.A. (2015) 'A Cyber-Physical Systems Architecture
   for Industry 4.0-Based Manufacturing Systems', Manufacturing Letters, Vol. 3,
   pp. 18–23.

8. Lei Y., Li N., Guo L., Li N., Yan T. and Lin J. (2018) 'Machinery Health
   Prognostics: A Systematic Review from Data Acquisition to RUL Prediction',
   Mechanical Systems and Signal Processing, Vol. 104, pp. 799–834.

9. Lee J., Qiu H., Yu G. and Lin J. (2007) 'Rexnord Technical Services: Bearing
   Data Set', NASA Ames Prognostics Data Repository, NASA Ames Research Center,
   Moffett Field, California.

10. Li X., Ding Q. and Sun J.Q. (2018) 'Remaining Useful Life Estimation in
    Prognostics Using Deep Convolution Neural Networks', Reliability Engineering
    and System Safety, Vol. 172, pp. 1–11.

11. Nectoux P., Gouriveau R., Medjaher K., Ramasso E., Chebel-Morello B.,
    Zerhouni N. and Varnier C. (2012) 'PRONOSTIA: An Experimental Platform for
    Bearings Accelerated Degradation Tests', Proceedings of the IEEE
    International Conference on Prognostics and Health Management, pp. 1–8.

12. Saxena A., Goebel K., Simon D. and Eklund N. (2008) 'Damage Propagation
    Modeling for Aircraft Engine Run-to-Failure Simulation', Proceedings of the
    International Conference on Prognostics and Health Management, pp. 1–9.

13. Shi W., Cao J., Zhang Q., Li Y. and Xu L. (2016) 'Edge Computing: Vision and
    Challenges', IEEE Internet of Things Journal, Vol. 3, No. 5, pp. 637–646.

14. Si X.S., Wang W., Hu C.H. and Zhou D.H. (2011) 'Remaining Useful Life
    Estimation — A Review on the Statistical Data Driven Approaches', European
    Journal of Operational Research, Vol. 213, No. 1, pp. 1–14.

15. Vaswani A., Shazeer N., Parmar N., Uszkoreit J., Jones L., Gomez A.N.,
    Kaiser L. and Polosukhin I. (2017) 'Attention Is All You Need', Advances in
    Neural Information Processing Systems, Vol. 30, pp. 5998–6008.

16. Wang B., Lei Y., Li N. and Li N. (2020) 'A Hybrid Prognostics Approach for
    Estimating Remaining Useful Life of Rolling Element Bearings', IEEE
    Transactions on Reliability, Vol. 69, No. 1, pp. 401–412.

17. Wu Y., Yuan M., Dong S., Lin L. and Liu Y. (2018) 'Remaining Useful Life
    Estimation of Engineered Systems Using Vanilla LSTM Neural Networks',
    Neurocomputing, Vol. 275, pp. 167–179.

18. Zheng S., Ristovski K., Farahat A. and Gupta C. (2017) 'Long Short-Term
    Memory Network for Remaining Useful Life Estimation', Proceedings of the
    IEEE International Conference on Prognostics and Health Management,
    pp. 88–95.
