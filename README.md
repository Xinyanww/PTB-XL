# Label provenance in PTB-XL: what a diagnostic label is a label *of*

Xinyan Wang · September 2026 · Python, ~ 4 hours of work

## Summary

PTB-XL is the most widely used public 12-lead ECG benchmark. About a third of the records in its official training folds carry machine-generated labels with no documented human validation, while its official validation and test folds contain none. I set out to ask whether this difference in how labels were produced changes the labels themselves.

I could not answer that question, and this repository documents why. Each design I tried failed for a specific, checkable reason: provenance is collinear with site, device, era, and case mix; a within-patient comparison removes the patient but not the selection into review; and in the closest discordant pair the two recordings are visibly different signals, so there was no stable patient state to disagree about. The last point turned out to be the most general one. A 10-second ECG is a sample of a process, and its label belongs to those ten seconds, not to the person.

The analysis ends on a possible way forward (Step 5) rather than a result.

## The question

A diagnostic label in a clinical dataset is the output of a process: a device records a signal, software proposes a reading, and sometimes a clinician confirms or revises it. Models trained on such labels learn whatever that process encodes. In PTB-XL the process is not uniform, so the natural question is whether labels produced without human review differ systematically from labels produced with it, holding the heart constant.

## Analysis path

Each step states what I asked, what I found, and why the finding does not settle the question. Numbers are from my run on the PTB-XL 1.0.3 metadata; rerunning the scripts reproduces them.

### Step 1: Describe provenance (`01_describe_provenance.py`)

Of 21,799 records (18,869 patients), 5,743 (26.3%) have no documented human validation. All 5,743 are machine-generated reports, and none has a `validated_by` entry. They make up 33.0% of folds 1–8 and 0% of folds 9–10, the folds PTB-XL recommends for validation and testing. So the dataset authors clearly separated them from evaluation, but models trained on the official split learn from a mixture of two labeling processes and are evaluated on only one.

The unreviewed records are not scattered: 5,741 of the 5,743 come from a single site on a single device model. Provenance is therefore not a record-level variable with usable variation. It is a block of the dataset.

### Step 2: Between-record comparison (`02_confounding.py`)

The block is labeled very differently from the rest: NORM 28.7% vs 49.4%, myocardial infarction 36.0% vs 20.9%. But it also differs in recording period, age, sex mix, and case mix, with a pacemaker rate roughly five times higher (3.2% vs 0.6%). A plainly sicker population explains a lower NORM rate without any appeal to the labeling process.

A logistic regression of NORM on block membership gives an odds ratio of about 0.4 with or without adjustment; adjustment does not shrink the gap. That coefficient is shown only to illustrate the problem. Block membership is collinear with site, device, era, and patients, and the available recording-quality flags move in inconsistent directions across the two groups, which suggests they were themselves annotated differently and are not valid controls. Holding the site fixed does not help either: the two devices at that site serve different populations (NORM 79% vs 29%, mean age roughly 40 vs 58).

### Step 3: Within-patient comparison (`03_within_patient.py`)

106 patients have at least one reviewed and one unreviewed record. Taking the closest pair for each, the median gap is about 11 days, the order is balanced, 99% of pairs are on the same device, and no pair shares a timestamp, so these are distinct recordings.

|                        | unreviewed not NORM | unreviewed NORM |
|------------------------|--------------------:|----------------:|
| **reviewed not NORM**  | 81                  | 2               |
| **reviewed NORM**      | 6                   | 17              |

Raw agreement (92%) is uninformative because most pairs are abnormal on both sides. Among the 25 patients where at least one record is NORM, 8 disagree (32%), and the direction is lopsided: 6 where only the reviewed record is NORM against 2 the other way. One of the 6 is a cross-device pair four years apart; excluding it gives 5 to 2. This is a small descriptive count, not an estimate.

The reviewed records in this comparison were selected for review, not sampled. Within the unreviewed site and device, the 335 reviewed records span the same years as the 5,741 unreviewed ones, so review followed some rule rather than a one-off batch. Reviewed records are older and less often NORM (20.6% vs 29.2%), and 29.9% of them received a second opinion against 2.5% dataset-wide. That selection pushes reviewed records toward looking *less* normal, opposite to the observed direction, so it cannot by itself produce the result. What I cannot rule out is that records were sent for review precisely because the machine reading looked wrong, in which case the 6 to 2 is that rule and not a human-versus-machine difference.

### Step 4: Look at the waveforms (`04_waveforms.py`)

![closest discordant pair](figures/closest_discordant_pair.png)

The closest discordant pair is one patient recorded twice, 63 seconds apart (ECG IDs 13530 and 13531). The earlier record carries rhythm and ischemia codes; the later one is labeled NORM. The two lead II tracings are plainly different to the eye: irregular R–R intervals and a large negative deflection in the first, a regular rhythm in the second.

So in this pair the labels are not two readings of the same signal. Either the rhythm changed within about a minute, or the first recording was poor and was repeated immediately; the script prints the recording-quality flags for both records to help decide. Either way, the within-patient design assumed a stable patient state between two recordings, and here that assumption fails at a gap of one minute.

### Step 5: A possible way forward (`05_edit_field_probe.py`, exploratory)

Several reviewed reports appear to keep the machine's original German text, including its "unconfirmed report" tag, and append the human revision after an `Edit:` marker. If that pattern holds, the machine reading and the human reading of the *same* recording are both recoverable from one field, with patient, moment, and device held fixed by construction. That is the comparison none of the steps above could make. The script checks whether the pattern is general (where `Edit:` appears, whether the machine tag survives review, whether the codes in the edit match the stored labels) and prints examples to read by hand. I have not yet analysed the edits themselves.

## What this does and does not show

It shows that provenance in PTB-XL is structured rather than random, that the training folds mix two labeling processes the test folds do not, and that the obvious comparisons cannot separate the labeling process from the patients. It does not show that human review changes labels, in either direction.

The broader point is about what the unit of a label is. Every design here, and most models trained on ECG labels, treats the label as a property of the patient. The 63-second pair suggests it is a property of a short recording, and that the variability between recordings is part of what any "label noise" analysis has to account for.

## Limitations

The within-patient sample is small (25 informative patients). "No documented human validation" is a statement about the metadata, and I have not verified from the original curation records whether these reports were never read or were read without being logged. SCP codes were derived from German free text; I have not examined how that mapping was done, and at least one machine report describes a T-wave abnormality while its stored code is NORM. I am not a clinician, and my reading of the tracings is limited to what is visible without training.

## Running the code

Download `ptbxl_database.csv` and `scp_statements.csv` from [PhysioNet PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) (open access) into `data/`. Waveforms are streamed record by record in Step 4, so the full archive is not needed.

```bash
pip install -r requirements.txt
python 01_describe_provenance.py
python 02_confounding.py
python 03_within_patient.py
python 04_waveforms.py        # needs internet access to PhysioNet
python 05_edit_field_probe.py
```

| File | Purpose |
|---|---|
| `common.py` | Loading, derived columns, within-patient pairing, waveform streaming |
| `01_describe_provenance.py` | How labels were produced, fold asymmetry, where unreviewed records come from |
| `02_confounding.py` | Why between-record comparisons cannot identify a provenance effect |
| `03_within_patient.py` | Within-patient 2×2, informative denominator, selection into review |
| `04_waveforms.py` | Discordant-pair figures and the 63-second pair |
| `05_edit_field_probe.py` | Exploratory check of the `Edit:` field |

Tables are written to `results/` and figures to `figures/`.
