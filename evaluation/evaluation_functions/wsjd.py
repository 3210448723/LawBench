import re
import os
import subprocess

"""
Task: legal document grammar correction
Metric: F0.5 score
文书校对
"""

# Resolve the utils directory relative to this file so the function works
# regardless of the caller's working directory.
_UTILS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils')
_UTILS_DIR = os.path.normpath(_UTILS_DIR)


def compute_wsjd(data_dict):
    origins, references, predictions = [], [], []
    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]
        if isinstance(question, list):
            question = question[0]['prompt']
        try:
            start = question.index('句子：\n') + 4
        except ValueError:
            # Fallback: use the whole prompt as the origin sentence
            start = 0
        origins.append(re.sub(r'\n|\t', '', question[start:].split('\n')[0]))
        # truncate predictions >5 tokens longer than the reference
        prediction = re.sub(r'\n|\t', '', prediction)
        if len(prediction) - len(answer) > 5:
            prediction = prediction[:len(answer) + 5]
        if len(prediction) == 0:
            prediction = "无内容"
        predictions.append(prediction)
        references.append(re.sub(r'\n|\t', '', answer))

    #generate input files for ChERRANT
    preds = [f'{i} \t {origin} \t {prediction} \n' for i, (origin, prediction) in enumerate(zip(origins, predictions))]
    golds = [f'{i} \t {origin} \t {reference} \n' for i, (origin, reference) in enumerate(zip(origins, references))]

    tmp_pred = os.path.join(_UTILS_DIR, 'tmp_pred.para')
    tmp_gold = os.path.join(_UTILS_DIR, 'tmp_gold.para')
    tmp_pred_m2 = tmp_pred + '.m2'
    tmp_gold_m2 = tmp_gold + '.m2'

    with open(tmp_pred, 'w') as f:
        f.writelines(preds)
    with open(tmp_gold, 'w') as f:
        f.writelines(golds)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # Run ChERRANT scripts from within _UTILS_DIR so their relative imports work
    orig_cwd = os.getcwd()
    try:
        os.chdir(_UTILS_DIR)
        subprocess.run(
            ['python3', 'parallel_to_m2.py', '-f', tmp_pred, '-o', tmp_pred_m2, '-g', 'char'],
            check=True,
        )
        subprocess.run(
            ['python3', 'parallel_to_m2.py', '-f', tmp_gold, '-o', tmp_gold_m2, '-g', 'char'],
            check=True,
        )
        output = subprocess.check_output(
            ['python3', 'compare_m2_for_evaluation.py', '-hyp', tmp_pred_m2, '-ref', tmp_gold_m2],
        )
    finally:
        os.chdir(orig_cwd)
        for tmp_file in (tmp_pred, tmp_gold, tmp_pred_m2, tmp_gold_m2):
            try:
                os.remove(tmp_file)
            except OSError:
                pass

    score = float(output.decode().split('\t')[-1].split('\n')[0])
    return {"score": score}
