def preprocess_sentence(text):
    sentences = [s.strip().lower() for s in text.split('.') if len(s) != 0]
    return sentences


def prediction_with_word_loc(predictions, sentences):
    result_with_iob = [list() for i in range(len(sentences))]
    for i, prediction in enumerate(predictions):
        start = 0
        end = 0
        for pred in prediction:
            word = list(pred.keys())[0]
            tag = pred[word]
            word = word.replace(".", "")

            start = sentences[i].index(word, end)

            size = len(word)
            end = start + size

            if tag != 'O':
                result_with_iob[i].append({'word': word, 'start': start, 'end': end, 'entity': tag})
    return result_with_iob


def prediction_with_ner(result_with_iob):
    result_with_ner = [list() for i in range(len(result_with_iob))]

    for i, results in enumerate(result_with_iob):
        for result in results:

            if 'I-' in result['entity']:
                if len(result_with_ner[i]) == 0:
                    result_with_ner[i].append({
                        'original': result['entity'],
                        'entity': result['entity'].split('I-')[1],
                        'start': result['start'],
                        'end': result['end'],
                        'word': result['word'],
                    })
                else:
                    last_result = result_with_ner[i].pop(-1)
                    if last_result['end'] == (result['start'] - 1):
                        result_with_ner[i].append({
                            'original': last_result['original'],
                            'entity': last_result['entity'],
                            'start': last_result['start'],
                            'end': result['end'],
                            'word': last_result['word'] + ' ' + result['word'],
                        })
                    else:
                        result_with_ner[i].append(last_result)
                        result_with_ner[i].append({
                            'original': result['entity'],
                            'entity': result['entity'].split('I-')[1],
                            'start': result['start'],
                            'end': result['end'],
                            'word': result['word'],
                        })
            else:
                result_with_ner[i].append({
                    'original': result['entity'],
                    'entity': result['entity'].split('B-')[1],
                    'start': result['start'],
                    'end': result['end'],
                    'word': result['word'],
                })
    return result_with_ner


def postprocess_predictions(predictions, sentences):
    result_with_iob = prediction_with_word_loc(predictions, sentences)
    result_with_ner = prediction_with_ner(result_with_iob)
    return result_with_ner
