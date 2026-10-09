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


# ============================================================
# FILE PATHS
# ============================================================

FILEPATH = 'structured_restaurant_data.json'
BACKUP_PATH = 'structured_restaurant_data.json.bak'

EXAMPLE_RESTAURANT_PARAGRAPH = '''Down in **Santa Monica**, **Mar de Cortez** serves as a **sun-drenched**, **casual taqueria** specializing in **Baja-style seafood**. With a **4.2/5** rating, it captures the salt-air energy of the coast through its signature beer-battered snapper tacos and zesty octopus ceviche, making it a premier spot for open-air dining near the pier. Price range:'''


# ============================================================
# DATA LOADING
# ============================================================

def load_data(file_path):
    """Load restaurant data from a JSON file."""

    if not os.path.exists(file_path):
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================
# DATA SAVING
# ============================================================

def save_data(data, file_path, backup_path):
    """Save restaurant data and create a backup of the old file."""

    if os.path.exists(file_path):
        shutil.copy2(file_path, backup_path)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# ============================================================
# RESTAURANT CARD
# ============================================================

def show_restaurant_card(data, index):
    """Display detailed information for a restaurant."""

    restaurant = data[index]

    print("\n--- Restaurant Details ---")

    for key, value in restaurant.items():
        print(f"{key}: {value}")


# ============================================================
# LLM PROMPT GENERATION
# ============================================================

def restaurant_data_structure_prompt_generation(restaurant_paragraph):
    """
    Generate a prompt for converting an unstructured restaurant
    paragraph into structured JSON.
    """

    prompt = f"""
Extract the restaurant information from the following paragraph
and return ONLY valid JSON.

Restaurant paragraph:
{restaurant_paragraph}

The JSON should contain the restaurant information using the same
structure and field names as the existing restaurant dataset.

Extract only information that is explicitly available in the paragraph.
Do not invent information.

Do not add explanations, markdown, or code fences.
Return only the JSON object.
"""

    return prompt


# ============================================================
# LLM MODEL
# ============================================================

def llm_model(system_msg, prompt_txt, params=None):
    """Send a prompt to IBM watsonx.ai."""

    model_id = "ibm/granite-4-h-small"
    project_id="skills-network"
    credentials = Credentials(url = "https://us-south.ml.cloud.ibm.com")

    model = ModelInference(
        model_id=model_id,
        credentials=credentials,
        project_id=project_id,
        params=params or {
            "decoding_method": "greedy",
            "max_new_tokens": 1000,
            "min_new_tokens": 1,
            "temperature": 0
        }
    )

    prompt = f"""
{system_msg}

{prompt_txt}
"""

    response = model.generate_text(prompt=prompt)

    return response


# ============================================================
# JSON AUTO REPAIR
# ============================================================

def JSON_auto_repair_prompts(response, error_message):
    """Generate a prompt to repair invalid JSON returned by the LLM."""

    prompt = f"""
The following response is supposed to be a valid JSON object,
but it contains a JSON formatting error.

Original response:
{response}

JSON parsing error:
{error_message}

Repair the JSON.

Return ONLY valid JSON.
Do not include markdown.
Do not include ```json.
Do not include any explanation.
"""

    return prompt


# ============================================================
# NEW DATA ENTRY PROCESS
# ============================================================

def new_data_entry_process(paragraph, itemId):
    """
    Convert a new restaurant paragraph into structured JSON
    using the LLM.
    """

    system_msg = """
You are a restaurant data extraction assistant.

Your task is to convert an unstructured restaurant description
into structured JSON data.

Extract only information that is supported by the input text.
Do not invent information.
Return valid JSON only.
"""

    prompt = restaurant_data_structure_prompt_generation(paragraph)

    response = llm_model(
        system_msg,
        prompt
    )

    try:
        restaurant_data = json.loads(response)

    except json.JSONDecodeError as e:

        repair_prompt = JSON_auto_repair_prompts(
            response,
            str(e)
        )

        repaired_response = llm_model(
            system_msg,
            repair_prompt
        )

        try:
            restaurant_data = json.loads(repaired_response)

        except json.JSONDecodeError:
            raise ValueError(
                "The LLM returned invalid JSON after automatic repair."
            )

    if isinstance(restaurant_data, dict):
        restaurant_data["itemId"] = itemId

    return restaurant_data


# ============================================================
# MAIN RESTAURANT MANAGEMENT UI
# ============================================================

def manage_restaurants(file_path, backup_path):

    while True:

        data = load_data(file_path)

        print(
            f"\n🏨 RESTAURANT DATABASE | Records: {len(data)}"
        )

        print("1. Browse All (Names)")
        print("2. View Detailed Record")
        print("3. Add New Restaurant")
        print("4. Edit Restaurant Info")
        print("5. Delete Restaurant")
        print("6. Exit")

        choice = input("\nAction: ")

        # ----------------------------------------------------
        # 1. BROWSE ALL RESTAURANTS
        # ----------------------------------------------------

        if choice == '1':

            print("\n--- Current Listings ---")

            for index, restaurant in enumerate(data):

                name = restaurant.get(
                    'name',
                    'N/A'
                )

                print(f"{index}: {name}")

        # ----------------------------------------------------
        # 2. VIEW DETAILED RECORD
        # ----------------------------------------------------

        elif choice == '2':

            try:

                index = int(
                    input("Enter record index: ")
                )

                if 0 <= index < len(data):

                    show_restaurant_card(
                        data,
                        index
                    )

                else:

                    print("Invalid index.")

            except ValueError:

                print("Invalid index.")

        # ----------------------------------------------------
        # 3, 4, 5. WRITE OPERATIONS
        # ----------------------------------------------------

        elif choice in ['3', '4', '5']:

            print(
                "\n❗ SECURITY WARNING: "
                "You are entering write-mode."
            )

            print(
                "Changes will be saved to the database immediately."
            )

            confirm = input(
                "Are you sure? (type 'yes' to proceed): "
            ).lower()

            if confirm != 'yes':

                print("Operation cancelled.")

                continue

            # ------------------------------------------------
            # 3. ADD NEW RESTAURANT
            # ------------------------------------------------

            if choice == '3':

                itemId = 1000000 + len(data) + 1

                description = input(
                    "\nEnter the new restaurant description:\n"
                )

                new_restaurant = new_data_entry_process(
                    description,
                    itemId
                )

                data.append(
                    new_restaurant
                )

                save_data(
                    data,
                    file_path,
                    backup_path
                )

                print("✅ Restaurant added.")

            # ------------------------------------------------
            # 4. EDIT RESTAURANT
            # ------------------------------------------------

            elif choice == '4':

                try:

                    index = int(
                        input(
                            "Enter record index to edit: "
                        )
                    )

                    if 0 <= index < len(data):

                        restaurant = data[index]

                        print("\n--- Edit Restaurant ---")

                        print(
                            "Press Enter to keep the current value."
                        )

                        for key in restaurant.keys():

                            current_value = restaurant[key]

                            new_value = input(
                                f"{key} [{current_value}]: "
                            )

                            if new_value.strip() != "":

                                restaurant[key] = new_value

                        save_data(
                            data,
                            file_path,
                            backup_path
                        )

                        print("✅ Record updated.")

                    else:

                        print("Invalid index.")

                except ValueError:

                    print("Invalid index.")

            # ------------------------------------------------
            # 5. DELETE RESTAURANT
            # ------------------------------------------------

            elif choice == '5':

                try:

                    index = int(
                        input(
                            "Enter record index to delete: "
                        )
                    )

                    if 0 <= index < len(data):

                        data.pop(index)

                        save_data(
                            data,
                            file_path,
                            backup_path
                        )

                        print("✅ Record deleted.")

                    else:

                        print("Invalid index.")

                except ValueError:

                    print("Invalid index.")

        # ----------------------------------------------------
        # 6. EXIT
        # ----------------------------------------------------

        elif choice == '6':

            print(
                "Exiting restaurant management system."
            )

            break

        # ----------------------------------------------------
        # INVALID OPTION
        # ----------------------------------------------------

        else:

            print("Invalid input.")


# ============================================================
# UNIT TESTS
# ============================================================

class TestRestaurantDatabase(unittest.TestCase):

    def setUp(self):
        """Create a temporary clean database for testing."""

        self.test_file = (
            'structured_restaurant_data_unit_test.json'
        )

        self.test_file_backup = (
            'structured_restaurant_data_unit_test.json.bak'
        )

        self.initial_data = [
            {
                "name": "Test Cafe",
                "location": "Test City"
            }
        ]

        with open(
            self.test_file,
            'w',
            encoding='utf-8'
        ) as f:

            json.dump(
                self.initial_data,
                f
            )

    def tearDown(self):
        """Clean up the test files after tests."""

        if os.path.exists(self.test_file):

            os.remove(self.test_file)

        if os.path.exists(self.test_file_backup):

            os.remove(self.test_file_backup)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_add_and_delete_restaurant_success(
        self,
        mock_stdout,
        mock_input
    ):
        """
        Test Scenario:
        Add a new restaurant and then delete it.
        """

        mock_restaurant = (
            'The Copper Sprout is a high-concept, Modern Appalachian '
            'farm-to-table destination that blends an industrial-chic '
            'aesthetic with rustic forest charm, featuring reclaimed wood '
            'and amber lighting to create a sophisticated yet cozy vibe. '
            'Priced in the $$ category, the menu celebrates seasonal '
            'foraging and local heritage, headlined by signature dishes '
            'like Cast-Iron Smoked Trout with pickled fiddlehead ferns '
            'and hand-foraged Wild Mushroom Risotto with aged goat cheese. '
            'The experience is designed to be intimate and earthy, making '
            'it a premier spot for those seeking high-quality, '
            'smokehouse-influenced cuisine in a refined, atmospheric setting.'
        )

        # ADD
        mock_input.side_effect = [
            '3',
            'yes',
            mock_restaurant,
            '6'
        ]

        try:

            manage_restaurants(
                self.test_file,
                self.test_file_backup
            )

        except SystemExit:

            pass

        with open(
            self.test_file,
            'r',
            encoding='utf-8'
        ) as f:

            data = json.load(f)

        print(data)

        self.assertEqual(
            len(data),
            2
        )

        self.assertIn(
            "Restaurant added.",
            mock_stdout.getvalue()
        )

        # DELETE
        mock_input.side_effect = [
            '5',
            'yes',
            '1',
            '6'
        ]

        try:

            manage_restaurants(
                self.test_file,
                self.test_file_backup
            )

        except SystemExit:

            pass

        with open(
            self.test_file,
            'r',
            encoding='utf-8'
        ) as f:

            data = json.load(f)

        print(data)

        self.assertEqual(
            len(data),
            1
        )

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_delete_security_cancel(
        self,
        mock_stdout,
        mock_input
    ):
        """
        Test Scenario:
        Try to delete a restaurant but cancel the operation.
        """

        mock_input.side_effect = [
            '5',
            'no',
            '6'
        ]

        manage_restaurants(
            self.test_file,
            self.test_file_backup
        )

        with open(
            self.test_file,
            'r',
            encoding='utf-8'
        ) as f:

            data = json.load(f)

        self.assertEqual(
            len(data),
            1
        )

        self.assertIn(
            "Operation cancelled.",
            mock_stdout.getvalue()
        )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # Run unit tests
    unittest.main()

    # To run the actual UI instead, comment out unittest.main()
    # and uncomment the following line:
    #
    # manage_restaurants(FILEPATH, BACKUP_PATH)
