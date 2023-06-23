from parascore import ParaScorer


def main():
    original = ["For a machine learning model to generate meaningful text, it must have a large amount of knowledge about the world as well as the ability to abstract."]
    candidate = ["the eye of the tiger"]
    para_scorer = ParaScorer(lang="en", model_type='bert-base-uncased', device='cuda')
    score = para_scorer.free_score(cands=candidate, sources=original, batch_size=16)[0]
    print(score.item())

if __name__ == '__main__':
    main()
