# %%
import os
import nbformat
import re
from nbformat.notebooknode import NotebookNode, from_dict
# %%


# notebook_path = "notebooks/coding_projects/P1_ANOVA/anova copy.ipynb"
# with open(notebook_path, 'r', encoding='utf-8') as f:
#     notebook = nbformat.read(f, as_version=4)
# notebook
# # with open(notebook_path, 'w', encoding='utf-8') as f:
# #     nbformat.write(notebook, f)
# type(notebook['cells'][0])
# notebook['cells'][0]
# NotebookNode?
# %%
def split_import_and_code_cells(notebook_path):
    """
    Process a Jupyter Notebook file, splitting cells with both import and non-import lines into two cells.
    The first new cell will contain only import statements, and the second will contain the rest of the code.
    """
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    new_cells = []

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            # Split the lines in the cell
            lines = cell["source"].splitlines()

            # Extract leading blank lines or lines starting with "#|"
            leading_lines = []
            while lines and (lines[0].strip() == "" or lines[0].startswith("#|")):
                leading_lines.append(lines.pop(0))

            # Separate import statements and other code lines
            import_lines = [
                line for line in lines if re.match(r"^\s*import\b|^\s*from\b", line)
            ]
            other_lines = [line for line in lines if line not in import_lines]

            if import_lines and other_lines:
                # Add the leading lines to the import cell
                new_cells.append(
                    from_dict(
                        {
                            "cell_type": "code",
                            "metadata": {},
                            "source": "\n".join(leading_lines + import_lines),
                            "outputs": [],
                        }
                    )
                )
                # Add the leading lines to the other code cell
                new_cells.append(
                    from_dict(
                        {
                            "cell_type": "code",
                            "metadata": {},
                            "source": "\n".join(leading_lines + other_lines),
                            "outputs": cell["outputs"],
                        }
                    )
                )
            else:
                # If no split is needed, retain the original cell
                new_cells.append(cell)
        else:
            # Retain non-code cells as is
            new_cells.append(cell)

    # Update the notebook with the modified cells
    notebook["cells"] = new_cells

    # Save the modified notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)


def process_notebooks_in_folder(folder_path):
    """
    Traverse all .ipynb files in a folder and apply the cell-splitting logic.
    """
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".ipynb"):
                notebook_path = os.path.join(root, file)
                print(f"Processing {notebook_path}")
                split_import_and_code_cells(notebook_path)


if __name__ == "__main__":
    # folder_path = input("Enter the path to the folder containing .ipynb files: ").strip()
    # folder_path = "notebooks/coding_projects/P1_ANOVA"
    folder_path = "notebooks"
    if os.path.isdir(folder_path):
        process_notebooks_in_folder(folder_path)
        print("Processing complete.")
    else:
        print("Invalid folder path.")
