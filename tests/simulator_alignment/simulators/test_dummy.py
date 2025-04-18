import csv
from unittest.mock import patch

from freezegun import freeze_time
import pytest

from simulator_alignment.data_models.sample import Sample
from simulator_alignment.simulators.dummy import CopyCatAssessor, RandomAssessor


@pytest.fixture
def sample_list():
    """Fixture to create sample list for testing, shared across test classes"""
    return [
        Sample(query="What is Paris?", passage="Paris is in France", groundtruth_relevance=3),
        Sample(query="How tall is Everest?", passage="Everest is tall", groundtruth_relevance=2),
        Sample(query="Who is Shakespeare?", passage="A famous writer", groundtruth_relevance=1),
        Sample(query="What is Python?", passage="Unrelated text", groundtruth_relevance=0),
    ]


class TestCopyCatAssessor:
    """Tests for the CopyCatAssessor class"""

    @pytest.fixture
    def assessor(self):
        """Fixture to create a CopyCatAssessor instance"""
        return CopyCatAssessor()

    def test_score_samples(self, assessor, sample_list):
        """Test that _score_samples copies groundtruth to predicted relevance"""
        result = assessor.assess(sample_list)
        assert result is sample_list

    def test_default_name_property(self, assessor):
        """Test the default name property"""
        assert assessor.name == "CopyCatAssessor"


class TestRandomAssessor:
    """Tests for the RandomAssessor class"""

    @pytest.fixture
    def default_assessor(self):
        """Fixture to create a RandomAssessor with default parameters"""
        return RandomAssessor()

    @pytest.fixture
    def custom_assessor(self):
        """Fixture to create a RandomAssessor with custom min/max values"""
        return RandomAssessor(min_val=1, max_val=5)

    def test_init_default_values(self, default_assessor):
        """Test initialization with default values"""
        assert default_assessor.min_val == 0
        assert default_assessor.max_val == 3

    def test_init_custom_values(self, custom_assessor):
        """Test initialization with custom values"""
        assert custom_assessor.min_val == 1
        assert custom_assessor.max_val == 5

    def test_score_samples_within_range(self, default_assessor, sample_list):
        """Test that _score_samples sets relevance within range"""
        with patch("random.randint", return_value=2):
            result = default_assessor._score_samples(sample_list)
            result[0].predicted_relevance == 2

    def test_custom_name_property(self, custom_assessor):
        """Test the custom name property"""
        assert custom_assessor.name == "RandomAssessor_1_5"

    def test_default_name_property(self, default_assessor):
        """Test the default name property"""
        assert default_assessor.name == "RandomAssessor_0_3"


class TestBaseSimulatorMethods:
    """Tests for inherited BaseSimulator methods"""

    @pytest.fixture
    def copycat_assessor(self):
        """Fixture to create a CopyCatAssessor instance"""
        return CopyCatAssessor()

    @pytest.fixture
    def random_assessor(self):
        """Fixture to create a RandomAssessor instance"""
        return RandomAssessor(min_val=0, max_val=3)

    def test_assess_without_file_output(self, copycat_assessor, sample_list):
        """Test assess method without file output"""
        result = copycat_assessor.assess(sample_list, write_to_file=False)

        assert result[0].predicted_relevance is not None
        assert result[0].predicted_relevance == result[0].groundtruth_relevance

    def test_assess_with_file_output(self, random_assessor, sample_list, tmp_path):
        """Test assess method with file output to specific path"""
        output_file = tmp_path / "test_output.tsv"

        with patch("random.randint", return_value=2):
            _ = random_assessor.assess(sample_list, write_to_file=True, output_path=output_file)

        assert output_file.exists()

    @freeze_time("2025-04-18 14:23:00")
    def test_write_to_file_auto_filename(self, copycat_assessor, sample_list, tmp_path):
        # cwd is tmp_path
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            _ = copycat_assessor.assess(sample_list, write_to_file=True)

        expected_filename = "CopyCatAssessor_results_2025-04-18T14-23-00.tsv"
        expected_path = tmp_path / "data" / "CopyCatAssessor" / expected_filename

        assert expected_path.exists()
        with expected_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            assert len(list(reader)) == len(sample_list)
