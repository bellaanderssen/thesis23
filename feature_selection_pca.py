import argparse
import helpers
import os
import sys

from datetime import datetime
from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection
from weka.filters import Filter


def select_attributes(data, searcher, evaluator, filename):
    attsel = AttributeSelection()
    attsel.search(searcher)
    attsel.evaluator(evaluator)
    start_time = datetime.now()
    attsel.select_attributes(data)
    end_time = datetime.now()

    # Filter for top PCA results
    pca_filter = Filter(
        classname="weka.filters.unsupervised.attribute.PrincipalComponents",
        options="-M 6".split())
    pca_filter.inputformat(data)
    filtered = pca_filter.filter(data)

    with open(f'{filename}.log', 'w') as f:
        f.write(f"Searcher: {searcher}")
        f.write(f"Evaluator: {evaluator}")
        f.write(f"Runtime (HH:mm:ss): {end_time - start_time}\n")
        f.write(attsel.results_string)
    helpers.save_csv(filtered, f"{filename}.csv")


parser = argparse.ArgumentParser()
parser.add_argument('infile')
parser.add_argument('--max-heap-size', default=None)
args = parser.parse_args()

try:
    filename = os.path.basename(args.infile)
    dirpath = os.path.dirname(args.infile)
    os.chdir(dirpath)

    with helpers.JVM(max_heap_size=args.max_heap_size):
        data = helpers.load_csv(filename)
        data = helpers.data_to_nominal(data)
        select_attributes(
            data,
            searcher=ASSearch(
                classname="weka.attributeSelection.Ranker"),
            evaluator=ASEvaluation(
                classname="weka.attributeSelection.PrincipalComponents"),
            filename="pca")

except Exception as e:
    with open('error.log', 'w') as f:
        f.write(str(e))
    sys.exit(1)
