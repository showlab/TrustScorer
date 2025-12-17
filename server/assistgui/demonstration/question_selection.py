# sample questions
import random
import json


def split_sentence(sentence):
    delimiters = ["is", "are", "were", "was", "does", "do", "did", "do", "does"]
    words = sentence.lower().split()
    result = []
    start = 0
    for i, word in enumerate(words):
        if word.lower() in delimiters:
            result.append(" ".join(words[start:i + 1]))
            start = i
            break
    result.append(" ".join(words[start + 1:]))
    return result


candidates = [
    'Where is the object skating?',
    'What colors are the object?',
    'Where are the object?',
    'Why are these household objects sitting at the side of the road?',
    'What number is the ump?',
    'Is it a object or object in the mirror?',
    'What color is the object that the man is wearing?',
    'Is the object watching TV?',
    'Is this object tidy?',
    'What are the yellow parts of this object called?',
    'How many objects does the creature have?',
    'Is the object about to enter a tunnel?',
    'What are the white objects in the bottom right corner?',
    'What color is the object in the object?',
    'About what temperature is illustrated here?'
]


def main():
    # uncomment the following line to load the candidates from a file
    # candidates = json.load(open('candidates.json'))
    query_type_dict = {}
    for candidate in candidates:
        query_type, content = split_sentence(candidate)
        query_type_dict.setdefault(query_type, []).append(candidate)

    # sample N keys of query_type_dict
    N = min(20, len(query_type_dict))
    query_types = random.sample(query_type_dict.keys(), N)

    # select M candidates for each query type
    selected_candidates = []
    for query_type in query_types:
        M = min(5, len(query_type_dict[query_type]))
        selected_candidates.extend(random.sample(query_type_dict[query_type], M))

    return selected_candidates


if __name__ == '__main__':
    selected_candidates = main()
    print(selected_candidates)
    # uncomment the following line to save the selected candidates to a file
    # json.dump(selected_candidates, open('selected_candidates.json', 'w'))
