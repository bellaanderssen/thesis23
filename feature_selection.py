import argparse
import helpers
import os
import sys

from configparser import ConfigParser
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

    # format selected attributes
    subset = ','.join(str(d+1) for d in attsel.selected_attributes)
    # remove all attributes not in subset
    remove = Filter(classname="weka.filters.unsupervised.attribute.Remove",
                    options=["-R", subset, "-V"])
    remove.inputformat(data)
    filtered = remove.filter(data)

    with open(f'{filename}.log', 'w') as f:
        f.write(f"Searcher: {searcher}")
        f.write(f"Seacher options: {searcher_options}\n")
        f.write(f"Evaluator: {evaluator}")
        f.write(f"Evaluator options: {evaluator_options}\n\n")
        f.write(f"Runtime (HH:mm:ss): {end_time - start_time}\n")
        f.write(attsel.results_string)
    helpers.save_csv(filtered, f"{filename}.csv")


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('indir')
parser.add_argument('--max-heap-size', default=None)
args = parser.parse_args()
config_file = helpers.assert_dir_contains_config(args.indir)

try:
    config = ConfigParser()
    config.optionxform = str  # preserve case in config keys
    config.read(config_file)
    os.chdir(args.indir)  # treat contents of file relative to config.ini
    data_filepath = config['meta']['data_path']
    helpers.assert_file_exists(data_filepath)

    with helpers.JVM(max_heap_size=args.max_heap_size):
        data = helpers.load_csv(data_filepath)
        data = helpers.data_to_nominal(data)

        for section in config:
            selector = config[section]
            searcher = selector.get('searcher', None)
            evaluator = selector.get('evaluator', None)
            if searcher is None or evaluator is None:
                continue
            searcher_options = selector.get('searcher_options', "")
            evaluator_options = selector.get('evaluator_options', "")
            select_attributes(
                data,
                searcher=ASSearch(
                    classname=searcher,
                    options=searcher_options.split()),
                evaluator=ASEvaluation(
                    classname=evaluator,
                    options=evaluator_options.split()),
                filename=section.replace(" ", "-")
            )

except Exception as e:
    with open('error.log', 'w') as f:
        f.write(str(e))
    sys.exit(1)
