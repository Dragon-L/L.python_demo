import numpy as np


def test_rank_candidates():
    questions = ['converting string to list', 'Sending array via Ajax fails']
    candidates = [['Convert Google results object (pure js) to Python object',
                   'C# create cookie from string and send it',
                   'How to use jQuery AJAX for an outside domain?'],
                  ['Getting all list items of an unordered list in PHP',
                   'WPF- How to update the changes in list item of a list',
                   'select2 not displaying search results']]
    results = [[(0, 'Convert Google results object (pure js) to Python object'),
                (1, 'C# create cookie from string and send it'),
                (2, 'How to use jQuery AJAX for an outside domain?')],
               [(0, 'Getting all list items of an unordered list in PHP'),
                (1, 'WPF- How to update the changes in list item of a list'),
                (2, 'select2 not displaying search results')]]
    for question, q_candidates, result in zip(questions, candidates, results):
        ranks = rank_candidates(question, q_candidates, wv_embeddings, 300)
        print(ranks)
        if not np.all(ranks == result):
            return "Check the function."
    return "Basic tests are passed."
