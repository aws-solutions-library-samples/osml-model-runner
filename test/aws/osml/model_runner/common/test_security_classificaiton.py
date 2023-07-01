#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import json
from dataclasses import asdict
from unittest import TestCase


class TestSecurityClassification(TestCase):
    def test_valid_classifications(self):
        from aws.osml.model_runner.common import Classification, ClassificationLevel

        c1 = Classification(level=ClassificationLevel.UNCLASSIFIED)
        assert c1.classification == "UNCLASSIFIED"

        c2 = Classification(level=ClassificationLevel.UNCLASSIFIED, releasability="For Official Use Only")
        assert c2.classification == "UNCLASSIFIED//FOR OFFICIAL USE ONLY"

        c3 = Classification(level=ClassificationLevel.SECRET, releasability="NOFORN")
        assert c3.classification == "SECRET//NOFORN"

        c4 = Classification(
            level=ClassificationLevel.TOP_SECRET,
            caveats=["FOO", "Bar", "BAZ"],
            releasability="ABC, DEF, GH",
        )
        assert c4.classification == "TOP SECRET//FOO/BAR/BAZ//ABC, DEF, GH"

    def test_invalid_classifications(self):
        from aws.osml.model_runner.common import Classification, ClassificationLevel, InvalidClassificationException

        with self.assertRaises(InvalidClassificationException):
            Classification(ClassificationLevel.UNCLASSIFIED, caveats=["FOO"])

        with self.assertRaises(InvalidClassificationException):
            Classification(ClassificationLevel.CONFIDENTIAL)

        with self.assertRaises(InvalidClassificationException):
            Classification(ClassificationLevel.TOP_SECRET, caveats=["FOO"])

        with self.assertRaises(InvalidClassificationException):
            Classification()

    def test_valid_classification_from_dict(self):
        from aws.osml.model_runner.common import Classification, ClassificationLevel, classification_asdict_factory

        level = ClassificationLevel.UNCLASSIFIED
        releasability = "For Official Use Only"
        c1 = Classification(level=level, releasability=releasability)
        c1_string = json.dumps(asdict(c1, dict_factory=classification_asdict_factory))

        c2 = Classification.from_dict(json.loads(c1_string))
        assert c2.level == level
        assert c2.releasability == releasability.upper()
        assert c2.classification == "UNCLASSIFIED//FOR OFFICIAL USE ONLY"
        assert c2.caveats is None

        level_2 = ClassificationLevel.TOP_SECRET
        caveats_2 = ["FOO", "Bar", "BAZ"]
        releasability_2 = "ABC, DEF, GH"
        c3 = Classification(level=level_2, caveats=caveats_2, releasability=releasability_2)
        c3_string = json.dumps(asdict(c3, dict_factory=classification_asdict_factory))
        c4 = Classification.from_dict(json.loads(c3_string))
        assert c4.level == level_2
        assert c4.caveats == [caveat.upper() for caveat in caveats_2]
        assert c4.releasability == releasability_2
        assert c4.classification == "TOP SECRET//FOO/BAR/BAZ//ABC, DEF, GH"

    def test_invalid_classification_from_dict(self):
        from aws.osml.model_runner.common import Classification, InvalidClassificationException

        with self.assertRaises(InvalidClassificationException):
            Classification.from_dict({"level": "TOP SECRET", "caveats": ["FOO"]})
