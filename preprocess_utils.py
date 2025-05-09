import re

def extract_murdough_case_text(full_text: str) -> str:
    """
    Extracts the relevant text from Murdough Center case files.
    The text is between "The Case:" and "Alternate Approaches and Survey Results".
    Returns an empty string if the section is not found or incomplete.
    """
    start_marker = "The Case:"
    # Regex to match "Alternate Approaches and Survey Results" with highly flexible separators
    # Allows for zero or more non-alphabetic characters between the main words.
    end_marker_regex = r"Alternate[^a-zA-Z]*Approaches[^a-zA-Z]*and[^a-zA-Z]*Survey[^a-zA-Z]*Results"

    start_index_marker = full_text.find(start_marker)
    if start_index_marker == -1:
        print(f"Warning: Start marker '{start_marker}' not found in Murdough case. Returning empty.")
        return ""

    # Content starts after the marker
    content_start_index = start_index_marker + len(start_marker)

    # Search for the end marker using regex from the content_start_index onwards, ignoring case
    content_to_search_end_marker = full_text[content_start_index:] # Store the slice for debugging
    end_match = re.search(end_marker_regex, content_to_search_end_marker, re.IGNORECASE)
    
    if not end_match:
        # debug_snippet = content_to_search_end_marker[:20000] # Get first 200 chars for the debug message
        print(f"Warning: Case-insensitive end marker regex '{end_marker_regex}' not found after start marker in Murdough case.")
        # print(f"         Searched in (first 20000 chars starting after start marker): '{debug_snippet}...'") # Added debug print
        print(f"         Returning empty as the specified section is incomplete.")
        return ""

    # The end_index for slicing should be the start of the matched end_marker, relative to content_start_index
    # Then add content_start_index to get the absolute index in full_text
    end_index_marker = content_start_index + end_match.start()

    return full_text[content_start_index:end_index_marker].strip()

def preprocess_content(file_content: str, source_name: str) -> str:
    """
    Preprocesses file content based on the source_name.
    `source_name` is expected to be the name of the input directory, lowercased for comparison.
    """
    if source_name.lower() == "murdoughcenter":
        return extract_murdough_case_text(file_content)
    # Add other source-specific preprocessing here if needed:
    # elif source_name.lower() == "another_source_type":
    #     return process_another_source(file_content)
    
    # Default: no preprocessing for other sources like "ethicsunwrapped"
    return file_content 