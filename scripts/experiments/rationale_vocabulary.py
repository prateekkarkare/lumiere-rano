"""
What do the radiologists actually write when they reason about new lesions?

Backs the vocabulary figures in docs/progress.html ("New lesions: still unsolved"). Their words
show the reasoning -- a 3-month clock and the radiation field. They are asking whether new
enhancement is tumour at all, because irradiated tissue enhances just as tumour does.

COHORT LOCK: rationales from the 17 held-out patients are excluded. When this count was first
run, inline, it read every patient in the ratings file and the held-out arm was not filtered out.
This script is the corrected version.

Run from the repository root.
"""
import csv
import json
import re

held_out = set(json.load(open('output/cohort/cohort_lock.json'))['held_out']['patient_ids'])
rows = list(csv.DictReader(open('LUMIERE-ExpertRating-v202211.csv')))
rr = next(k for k in rows[0] if k.startswith('Rating rationale'))
texts = [r[rr].strip() for r in rows if r[rr].strip() and r['Patient'] not in held_out]
n_pat = len({r['Patient'] for r in rows if r[rr].strip() and r['Patient'] not in held_out})
print(f"{len(texts)} non-empty rationales from {n_pat} patients (held-out arm excluded)\n")

FIELD_OR_CLOCK = re.compile(r'irradiat|radiation|field|months', re.I)
for phrase in ['less than 3 months', 'irradiat', 'field', 'pseudo', 'new target',
               'new non-target', 'new non-measurable', 'T2-Progr']:
    print(f"  {phrase:<22} {sum(1 for t in texts if phrase.lower() in t.lower()):>4}")
n_new = sum(1 for t in texts if 'new' in t.lower())
n_fc = sum(1 for t in texts if FIELD_OR_CLOCK.search(t))
n_both = sum(1 for t in texts if 'new' in t.lower() and FIELD_OR_CLOCK.search(t))
print(f"\n  mention a new lesion:               {n_new}/{len(texts)}")
print(f"  invoke the field or the 3-month clock: {n_fc}/{len(texts)} ({100*n_fc/len(texts):.0f}%)")
print(f"  both:                               {n_both}")
