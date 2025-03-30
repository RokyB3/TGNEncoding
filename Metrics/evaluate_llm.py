def hits_at_1(predictions, ground_truths):
    assert len(predictions) == len(ground_truths)
    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        pred = pred.strip().lower()
        gt = gt.strip().lower()
        correct += int(pred == gt)
    return correct / len(predictions)

def exact_match(predictions, ground_truths):
    return hits_at_1(predictions, ground_truths)  # For string-based answers
