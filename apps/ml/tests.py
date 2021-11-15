from django.test import TestCase

from apps.ml.cluster_classifier.random_forest import RandomForestClassifier
import inspect
from apps.ml.registry import MLRegistry
class MLTests(TestCase):
    def test_rf_algorithm(self):
        input_data = {'Unnamed: 0':0,
            'chalkiness': 2.3225,
            'average_length':7.06,
            'average_width': 2.37,
            'raw_grain_length':7.525,
            'raw_grain_width': 2.325,
            'raw_grain_shape':3.25,
            'cooked_grain_length':12.675,
            'cooked_grain_width': 3.7,
            'cooked_grain_shape':3.4,
            'amylose_content': 18.35,
            'weight_brown_rice':95.7,
            'percent_brown_rice':76.55,
            'weight_milled_rice':83.3,
            'percent_milled_rice': 66.675,
            'weight_head_rice': 57.5,
            'percent_head_rice': 46.0
            }
        my_alg = RandomForestClassifier()
        response = my_alg.compute_prediction(input_data)
        self.assertEqual('OK', response['status'])
        self.assertTrue('label' in response)
        self.assertEqual('<=50K', response['label'])


# ...
# the rest of the code
# ...

# add below method to MLTests class:
    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "cluster_classifier"
        algorithm_object = RandomForestClassifier()
        algorithm_name = "random forest"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "Vasu"
        algorithm_description = "Random Forest with simple pre- and post-processing"
        algorithm_code = inspect.getsource(RandomForestClassifier)
        # add to registry
        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                    algorithm_status, algorithm_version, algorithm_owner,
                    algorithm_description, algorithm_code)
        # there should be one endpoint available
        self.assertEqual(len(registry.endpoints), 1)