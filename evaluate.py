import argparse
import helpers
import os
import sys

from configparser import ConfigParser
from datetime import datetime
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random


def evaluate(data, classifier):
    start_time = datetime.now()
    classifier.build_classifier(data)
    end_time = datetime.now()
    build_time = end_time - start_time

    evaluation = Evaluation(data)
    start_time = datetime.now()
    evaluation.crossvalidate_model(classifier, data, 10, Random(42))
    end_time = datetime.now()
    evaluation_time = end_time - start_time

    with open(f'{experiment_name}.log', 'w') as f:
        f.write(f"Classifier: {classname}\n")
        f.write(f"Options: {options if options else 'default'}\n")
        f.write(f"Classifier build time (HH:mm:ss): {build_time}\n")
        f.write(f"Evaluation time (HH:mm:ss): {evaluation_time}\n")
        f.write(f"\n")
        f.write(f"{classifier}\n")
        f.write(f"\n")
        f.write(f"{evaluation.summary()}\n")
        f.write(f"{evaluation.class_details()}\n")
        f.write(f"{evaluation.confusion_matrix}\n")
        f.write("\n")
        f.write(evaluation_class_summary(evaluation, 0))
        f.write(evaluation_class_summary(evaluation, 1))


def evaluation_class_summary(evaluation, class_index):
    return (
        f"Class {class_index} details\n"
        f"Area under ROC: {evaluation.area_under_roc(class_index)}\n"
        f"False negatives: {evaluation.num_false_negatives(class_index)}\n"
        f"True negatives: {evaluation.num_true_negatives(class_index)}\n"
        f"False positives: {evaluation.num_false_positives(class_index)}\n"
        f"True positives: {evaluation.num_true_positives(class_index)}\n\n"
    )


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('indir', help="""
    A directory containing a config.ini file. All classifier results will be
    output to this file.
    """)
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
        data = helpers.fill_na(data)
        if config['meta'].get('data_is_numeric'):
            # Only convert the class to nominal
            data = helpers.data_to_nominal(data, filter_options="-R last")
        else:
            # Convert all columns to nominal
            data = helpers.data_to_nominal(data)

        for section in config:
            experiment_name = section
            # skip experiment if it has previously run
            if os.path.exists(f'{experiment_name}.log'):
                continue
            experiment = config[section]
            classname = experiment.get('classname', None)
            if classname is None:
                continue
            options = experiment.get('options', "")
            split_string = r' \\ ' if r' \\ ' in options else ' '
            classifier = Classifier(
                classname=classname,
                options=options.split(split_string))
            evaluate(data, classifier)

except Exception as e:
    with open('error.log', 'w') as f:
        f.write(str(e))
    sys.exit(1)
