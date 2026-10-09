from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
import json
import os
import shutil
import io
import unittest
from unittest.mock import patch

FILEPATH = 'structured_restaurant_data.json'
BACKUP_PATH = 'structured_restaurant_data.json.bak'
EXAMPLE_RESTAURANT_PARAGRAPH = (
    'Down in **Santa Monica**, **Mar de Cortez** serves as a **sun-drenched**, '
    '**casual taqueria** specializing in **Baja-style seafood**. With a **4.2/5** rating, '
    'it captures the salt-air energy of the coast through its signature beer-battered '
    'snapper tacos and zesty octopus ceviche, making it a premier spot for open-air '
    'dining near the pier. Price range: $$'
)

# ======================================================================
# Schema (match this to the schema you defined in Lesson 1)
# ======================================================================
class Restaurant(BaseModel):
    id: Optional[int] = None
    name: str
    neighborhood: str
    cuisine: str
    rating: float = Field(ge=0, le=5)
    price_range: str
    vibes: List[str]
    signature_dishes: List[str]
    description: str


EXAMPLE_RESTAURANT_JSON = {
    "id": 1000000,
    "name": "Mar de Cortez",
    "neighborhood": "Santa Monica",
    "cuisine": "Baja-style seafood",
    "rating": 4.2,
    "price_range": "$$",
    "vibes": ["sun-drenched", "casual", "open-air"],
    "signature_dishes": ["beer-battered snapper tacos", "octopus ceviche"],
    "description": "A sun-drenched, casual taqueria near the pier specializing in Baja-style seafood."
}

SYSTEM_MSG = (
    "You are a precise data-structuring assistant. You convert restaurant "
    "descriptions into valid JSON that follows the given schema exactly. "
    "Respond with JSON only: no explanations and no markdown."
)

# ======================================================================
# Exercise 1: LLM integration
# ======================================================================
def restaurant_data_structure_prompt_generation(restaurant_paragraph):
    """Build the prompt that asks the LLM to structure a restaurant paragraph."""
    schema = json.dumps(Restaurant.model_json_schema(), indent=2)
    example_output = json.dumps(EXAMPLE_RESTAURANT_JSON, indent=2)

    return f"""Convert the restaurant description below into a single JSON object.

Follow this JSON schema exactly:
{schema}

Rules:
- Return ONLY the JSON object, with no markdown fences or extra text.
- "rating" must be a number between 0 and 5.
- "price_range" must be one of "$", "$$", "$$$", "$$$$".
- "vibes" and "signature_dishes" must be lists of short strings.
- Use only information found in the description. Do not invent details.
- Set "id" to null; it will be assigned later.

Example input:
{EXAMPLE_RESTAURANT_PARAGRAPH}

Example output:
{example_output}

Now convert this description:
{restaurant_paragraph}

JSON output:"""


# Granite is used because it is a small, cheap, fast model that is
# accurate enough for structured extraction tasks like this one.
_MODEL = None

def _get_model():
    """Create the model once and reuse it."""
    global _MODEL
    if _MODEL is None:
        credentials = Credentials(url="https://us-south.ml.cloud.ibm.com")
        project_id = (
            os.environ.get("WATSONX_AI_PROJECT_ID")
            or os.environ.get("WATSONX_PROJECT_ID")
            or "skills-network"
        )
        _MODEL = ModelInference(
            model_id="ibm/granite-4-h-small",
            credentials=credentials,
            project_id=project_id,
        )
    return _MODEL


def llm_model(system_msg, prompt_txt, params=None):
    """Send a system message and prompt to the LLM and return the text reply."""
    default_params = {"temperature": 0, "max_tokens": 1024}
    if params:
        default_params.update(params)

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt_txt},
    ]
    response = _get_model().chat(messages=messages, params=default_params)
    return response["choices"][0]["message"]["content"]


def JSON_auto_repair_prompts(response, error_message):
    """Build a prompt asking the LLM to fix its own invalid JSON output."""
    schema = json.dumps(Restaurant.model_json_schema(), indent=2)
    return f"""The JSON below is invalid or does not match the required schema.

Invalid output:
{response}

Error message:
{error_message}

Required schema:
{schema}

Fix the JSON so that it is valid and matches the schema. Keep all the original
information. Return ONLY the corrected JSON object, with no markdown fences or
extra text."""


def _extract_json(text):
    """Strip markdown fences and surrounding text, keeping just the JSON object."""
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def new_data_entry_process(paragraph, itemId, max_retries=3):
    """Structure a new restaurant paragraph into a validated dict.

    Returns the structured record (with its id set), or None if the LLM
    could not produce valid JSON after the allowed repair attempts.
    """
    prompt = restaurant_data_structure_prompt_generation(paragraph)
    response = llm_model(SYSTEM_MSG, prompt)

    for attempt in range(max_retries + 1):
        candidate = _extract_json(response)
        try:
            record = Restaurant.model_validate_json(candidate)
            record.id = itemId
            return record.model_dump()
        except (ValidationError, ValueError) as e:
            if attempt == max_retries:
                print(f"❌ Could not structure the paragraph after {max_retries} repair attempts: {e}")
                return None
            print(f"⚠️ Invalid JSON (attempt {attempt + 1}), asking the LLM to repair it...")
            repair_prompt = JSON_auto_repair_prompts(candidate, str(e))
            response = llm_model(SYSTEM_MSG, repair_prompt)

# ======================================================================
# File helpers
# ======================================================================
def load_data(file_path):
    """Load the restaurant list from disk; return [] if missing or unreadable."""
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("⚠️ Data file is not valid JSON. Starting with an empty list.")
        return []


def save_data(file_path, data, backup_path=None):
    """Save the data, first copying the existing file to a backup."""
    if backup_path and os.path.exists(file_path):
        shutil.copy2(file_path, backup_path)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def show_restaurant_card(res, index):
    """Print one restaurant record as a readable card."""
    print(f"\n--- Restaurant #{index} ---")
    for key, value in res.items():
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        print(f"{key}: {value}")


def _read_index(data, prompt):
    """Ask for a 1-based record number; return a 0-based index or None."""
    raw = str(input(prompt)).strip()
    if raw.isdigit() and 1 <= int(raw) <= len(data):
        return int(raw) - 1
    print("Invalid index.")
    return None

# ======================================================================
# Command-line UI
# ======================================================================
def manage_restaurants(file_path, backup_path):
    while True:
        data = load_data(file_path)
        print(f"\n🏨 RESTAURANT DATABASE | Records: {len(data)}")
        print("1. Browse All (Names)")
        print("2. View Detailed Record")
        print("3. Add New Restaurant")
        print("4. Edit Restaurant Info")
        print("5. Delete Restaurant")
        print("6. Exit")

        choice = input("\nAction: ")

        if choice == '1':
            print("\n--- Current Listings ---")
            for i, res in enumerate(data, start=1):
                print(f"{i}. {res.get('name', 'N/A')}")

        elif choice == '2':
            idx = _read_index(data, "Enter record number: ")
            if idx is not None:
                show_restaurant_card(data[idx], idx + 1)
            continue

        elif choice in ['3', '4', '5']:
            # Strict Security Warning
            print("\n❗ SECURITY WARNING: You are entering write-mode.")
            print("Changes will be saved to the database immediately.")
            confirm = input("Are you sure? (type 'yes' to proceed): ").lower()
            if confirm != 'yes':
                print("Operation cancelled.")
                continue

            if choice == '3':  # ADD NEW DATA
                itemId = 1000000 + len(data) + 1  # the item id for the new data

                paragraph = input("Enter the new restaurant description: ")
                new_record = new_data_entry_process(paragraph, itemId)
                if new_record is None:
                    print("❌ Restaurant not added.")
                    continue
                data.append(new_record)
                save_data(file_path, data, backup_path)
                print("✅ Restaurant added.")

            elif choice == '4':  # EDIT DATA
                idx = _read_index(data, "Enter record number to edit: ")
                if idx is None:
                    continue
                record = data[idx]
                print("Press Enter to keep the current value.")
                for key, current in list(record.items()):
                    if key == "id":
                        continue
                    new_val = input(f"{key} [{current}]: ").strip()
                    if not new_val:
                        continue
                    if isinstance(current, list):
                        record[key] = [v.strip() for v in new_val.split(",")]
                    elif isinstance(current, bool):
                        record[key] = new_val.lower() in ("true", "yes", "1")
                    elif isinstance(current, (int, float)):
                        try:
                            record[key] = type(current)(new_val)
                        except ValueError:
                            print(f"Invalid number for '{key}', keeping the old value.")
                    else:
                        record[key] = new_val
                save_data(file_path, data, backup_path)
                print("✅ Record updated.")
                continue

            elif choice == '5':  # DELETE DATA
                idx = _read_index(data, "Enter record number to delete: ")
                if idx is None:
                    continue
                removed = data.pop(idx)
                save_data(file_path, data, backup_path)
                print(f"✅ Deleted '{removed.get('name', 'N/A')}'.")
                continue

        elif choice == '6':  # EXIT
            break
        else:
            print("Invalid input.")

# ======================================================================
# Unit tests
# ======================================================================
class TestRestaurantDatabase(unittest.TestCase):

    def setUp(self):
        """Create a temporary clean database for testing."""
        self.test_file = 'structured_restaurant_data_unit_test.json'
        self.test_file_backup = 'structured_restaurant_data_unit_test.json.bak'
        self.initial_data = [{"name": "Test Cafe", "location": "Test City"}]
        with open(self.test_file, 'w') as f:
            json.dump(self.initial_data, f)

    def tearDown(self):
        """Clean up the test files after tests."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.test_file_backup):
            os.remove(self.test_file_backup)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    @patch('restaurant_data_management.llm_model')
    def test_add_and_delete_restaurant_success(self, mock_llm, mock_stdout, mock_input):
        """
        Test Scenario: Add a new restaurant, then delete one.
        Inputs: '3' (Add), 'yes' (Confirm), paragraph, '6' (Exit)
        """
        mock_llm.return_value = json.dumps({
            "name": "The Copper Sprout",
            "neighborhood": "Appalachia",
            "cuisine": "Modern Appalachian",
            "rating": 4.5,
            "price_range": "$$",
            "vibes": ["cozy", "rustic"],
            "signature_dishes": ["Cast-Iron Smoked Trout", "Wild Mushroom Risotto"],
            "description": "A high-concept farm-to-table destination."
        })
        mock_restaurant = 'The Copper Sprout is a high-concept, Modern Appalachian farm-to-table destination that blends an industrial-chic aesthetic with rustic forest charm, featuring reclaimed wood and amber lighting to create a sophisticated yet cozy vibe. Priced in the $$ category, the menu celebrates seasonal foraging and local heritage, headlined by signature dishes like Cast-Iron Smoked Trout with pickled fiddlehead ferns and hand-foraged Wild Mushroom Risotto with aged goat cheese. The experience is designed to be intimate and earthy, making it a premier spot for those seeking high-quality, smokehouse-influenced cuisine in a refined, atmospheric setting.'
        mock_input.side_effect = ['3', 'yes', mock_restaurant, '6']

        manage_restaurants(self.test_file, self.test_file_backup)

        with open(self.test_file, 'r') as f:
            data = json.load(f)

        print(data)
        self.assertEqual(len(data), 2)
        self.assertIn("✅ Restaurant added.", mock_stdout.getvalue())

        mock_input.side_effect = ['5', 'yes', 1, '6']

        manage_restaurants(self.test_file, self.test_file_backup)

        with open(self.test_file, 'r') as f:
            data = json.load(f)

        print(data)
        self.assertEqual(len(data), 1)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_delete_security_cancel(self, mock_stdout, mock_input):
        """
        Test Scenario: Try to delete but say 'no' to the security warning.
        Inputs: '5' (Delete), 'no' (Cancel), '6' (Exit)
        """
        mock_input.side_effect = ['5', 'no', '6']

        manage_restaurants(self.test_file, self.test_file_backup)

        with open(self.test_file, 'r') as f:
            data = json.load(f)

        self.assertEqual(len(data), 1)  # Data should remain unchanged
        self.assertIn("Operation cancelled.", mock_stdout.getvalue())


if __name__ == "__main__":
    unittest.main()  # Unit Test
    # manage_restaurants(FILEPATH, BACKUP_PATH)  # Actual UI Call