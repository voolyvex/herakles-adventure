import os
import re
import shutil
import sys

# Configurable paths
SOURCE_MD = "bullfinchs-mythology.md"
CHUNKS_DIR = "lore_chunks"

# Clean up old chunks
if os.path.exists(CHUNKS_DIR):
    shutil.rmtree(CHUNKS_DIR)
os.makedirs(CHUNKS_DIR, exist_ok=True)

# Read the whole file
with open(SOURCE_MD, encoding="utf-8") as f:
    lines = f.readlines()

def is_chapter_heading(line):
    # Matches lines like 'CHAPTER III.' or 'CHAPTER X' etc.
    return bool(re.match(r"^\s*CHAPTER [IVXLCDM]+\.?\s*$", line.strip()))

def is_story_title(line):
    # Matches lines that are mostly uppercase, potentially with 'AND' or '-', and longer than 5 chars
    # Excludes lines that are just 'CHAPTER...' even if uppercase
    stripped = line.strip()
    if is_chapter_heading(stripped):
        return False
    # Improved regex: allows for lowercase 'and', requires more than one word usually unless short like 'IO'
    if len(stripped) > 3 and (re.match(r"^[A-Z][A-Z\s,'\-&]+[A-Z.]?$", stripped) or re.match(r"^[A-Z][a-z]+ AND [A-Z][a-z]+$", stripped)):
        # Avoid matching lines that are likely just part of the text (e.g., full sentences in caps)
        if len(stripped.split()) > 6: # Heuristic: Titles usually aren't very long sentences
             return False
        return True
    # Allow shorter single names if fully uppercase and potentially followed by a period or nothing
    if len(stripped) >= 2 and re.match(r"^[A-Z][A-Z.]*$", stripped):
         # Avoid matching common acronyms or single uppercase words within text
         if len(stripped) > 1 and stripped != "I": # Exclude single 'I'
              return True
    return False

def is_title_page(line):
    # Matches the title page heading
    return bool(re.match(r"^\s*1855\s*$", line.strip()))

def should_combine_with_previous(chunk_lines):
    """Determine if this chunk should be combined with the previous one"""
    # If it's just a title page or introduction, combine it
    if len(chunk_lines) <= 2:
        return True
    # If it's a very short chunk (less than 5 lines), combine it
    if len(chunk_lines) < 5:
        return True
    return False

# Identify all chapter and story titles as potential split points
chunk_starts = []
last_start = -1
for i, line in enumerate(lines):
    # Treat both chapter headings and story titles as start points for chunks
    if (is_chapter_heading(line) or is_story_title(line)) and line.strip():
        # Avoid adding start points too close together (e.g. title right after chapter heading)
        # Also ensure the line isn't just noise (e.g. extremely short) unless it's a known short title pattern
        if last_start == -1 or (i > last_start + 1) or (is_chapter_heading(lines[last_start])):
             chunk_starts.append(i)
             last_start = i
        # Simple heuristic to potentially replace a less likely title with a more likely one immediately following
        elif not is_chapter_heading(lines[last_start]):
             if len(line.strip()) > len(lines[last_start].strip()):
                 chunk_starts[-1] = i # Replace previous start point
                 last_start = i

# Ensure the very beginning is a start point if missed
if 0 not in chunk_starts:
     chunk_starts.insert(0, 0)

# Add end-of-file as the final boundary
if len(lines) not in chunk_starts:
    chunk_starts.append(len(lines))

# Process chunks
final_chunks = []
current_chunk = []
for i in range(len(chunk_starts) - 1):
    start = chunk_starts[i]
    end = chunk_starts[i+1]
    chunk_lines = lines[start:end]
    
    # Combine with previous chunk if it's too short
    if should_combine_with_previous(chunk_lines):
        if current_chunk:
            current_chunk.extend(chunk_lines)
    else:
        if current_chunk:
            final_chunks.append(current_chunk)
        current_chunk = chunk_lines

# Add the last chunk
if current_chunk:
    final_chunks.append(current_chunk)

# Write the refined chunks to files
chunk_count = 0
for idx, chunk_lines in enumerate(final_chunks):
    # Generate a safe filename from the first non-empty line (title)
    safe_title = None
    for l in chunk_lines:
        line_content = l.strip()
        if line_content:
            # Use the identified title directly
            safe_title = re.sub(r'[^a-zA-Z0-9]+', '_', line_content).strip('_')[:50]
            break
    if not safe_title:
        safe_title = f"chunk_{chunk_count+1}" # Fallback

    # Ensure unique filenames if titles clash
    filename_base = f"{chunk_count+1:03d}_{safe_title}"
    filename = filename_base + ".md"
    counter = 1
    while os.path.exists(os.path.join(CHUNKS_DIR, filename)):
        filename = f"{filename_base}_{counter}.md"
        counter += 1

    out_path = os.path.join(CHUNKS_DIR, filename)
    try:
        with open(out_path, "w", encoding="utf-8") as outf:
            outf.writelines(chunk_lines)
        chunk_count += 1
    except Exception as e:
        print(f"Error writing file {filename}: {e}")


# Analysis of chunk lengths
chunk_lengths = []
if chunk_count > 0:
    for fname in os.listdir(CHUNKS_DIR):
        if fname.endswith('.md'):
            try:
                with open(os.path.join(CHUNKS_DIR, fname), encoding='utf-8') as f:
                     chunk_lengths.append(len(f.readlines()))
            except Exception as e:
                print(f"Error reading {fname} for analysis: {e}")

print(f"Chunking complete. Created {chunk_count} refined chunks.")
if chunk_lengths:
    try:
        print(f"Min/Max/Avg chunk line count: {min(chunk_lengths)}, {max(chunk_lengths)}, {sum(chunk_lengths)//len(chunk_lengths)}")
    except ValueError: # Handles case where chunk_lengths might be empty despite chunk_count > 0 if reading fails
        print("Could not analyze chunk lengths.")
else:
    print("No chunks were created or analysis failed.")

sys.exit(0)

# Clean up old chunks
if os.path.exists(CHUNKS_DIR):
    shutil.rmtree(CHUNKS_DIR)
os.makedirs(CHUNKS_DIR, exist_ok=True)

# Read the whole file
with open(SOURCE_MD, encoding="utf-8") as f:
    lines = f.readlines()

def is_chapter_heading(line):
    # Matches lines like 'CHAPTER III.' or 'CHAPTER X' etc.
    return bool(re.match(r"^\s*CHAPTER [IVXLCDM]+\.?\s*$", line.strip()))

def is_story_title(line):
    # Matches lines that are mostly uppercase, potentially with 'AND' or '-', and longer than 5 chars
    # Excludes lines that are just 'CHAPTER...' even if uppercase
    stripped = line.strip()
    if is_chapter_heading(stripped):
        return False
    # Improved regex: allows for lowercase 'and', requires more than one word usually unless short like 'IO'
    if len(stripped) > 5 and (re.match(r"^[A-Z][A-Z\s,'\-&]+[A-Z.]?$", stripped) or re.match(r"^[A-Z][a-z]+ AND [A-Z][a-z]+$", stripped)):
        return True
    # Allow shorter single names if fully uppercase
    if len(stripped) <= 5 and re.match(r"^[A-Z]+$", stripped):
        return True
    return False

# Identify all chapter and story titles as potential split points
chunk_starts = []
last_start = -1
for i, line in enumerate(lines):
    # Treat both chapter headings and story titles as start points for chunks
    if (is_chapter_heading(line) or is_story_title(line)) and line.strip():
        # Avoid adding start points too close together (e.g. title right after chapter heading)
        if last_start == -1 or (i > last_start + 1):
             chunk_starts.append(i)
             last_start = i
        elif not is_chapter_heading(lines[last_start]): # If previous was not chapter, maybe replace it if this is better title
             # Heuristic: Prefer longer titles if they appear right after short ones
             if len(line.strip()) > len(lines[last_start].strip()):
                 chunk_starts[-1] = i # Replace previous start point
                 last_start = i

# Ensure the very beginning is a start point if missed
if 0 not in chunk_starts:
     chunk_starts.insert(0, 0)

# Add end-of-file as the final boundary
if len(lines) not in chunk_starts:
    chunk_starts.append(len(lines))

# Create chunks based on identified start points
final_chunks = []
for i in range(len(chunk_starts) - 1):
    start = chunk_starts[i]
    end = chunk_starts[i+1]
    chunk_lines = lines[start:end]
    # Keep the chunk if it has more than just whitespace or a single title line
    non_empty_lines = [l for l in chunk_lines if l.strip()]
    if len(non_empty_lines) > 1:
        final_chunks.append(chunk_lines)
    elif len(non_empty_lines) == 1 and (i + 2 < len(chunk_starts)):
        # If this chunk is just a title and the *next* one has content, merge title here
        next_start = chunk_starts[i+1]
        next_end = chunk_starts[i+2]
        next_chunk_lines = lines[next_start:next_end]
        if len([l for l in next_chunk_lines if l.strip()]) > 0:
             final_chunks.append(chunk_lines) # Keep the title-only chunk before content chunk

# Write the refined chunks to files
chunk_count = 0
for idx, chunk_lines in enumerate(final_chunks):
    # Generate a safe filename from the first non-empty line (title)
    safe_title = None
    title_line_index = -1
    for k, l in enumerate(chunk_lines):
        if l.strip():
            title_line_index = k
            # Remove potential chapter part for cleaner filenames
            title_text = re.sub(r"^CHAPTER [IVXLCDM]+\.\s*", "", l.strip())
            safe_title = re.sub(r'[^a-zA-Z0-9]+', '_', title_text).strip('_')[:50]
            break # Found first non-empty line
    if not safe_title:
        safe_title = f"chunk_{chunk_count+1}" # Fallback

    # Ensure unique filenames if titles clash
    filename_base = f"{chunk_count+1:03d}_{safe_title}"
    filename = filename_base + ".md"
    counter = 1
    while os.path.exists(os.path.join(CHUNKS_DIR, filename)):
        filename = f"{filename_base}_{counter}.md"
        counter += 1

    out_path = os.path.join(CHUNKS_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as outf:
        # If first line was chapter heading, skip it if a story title follows immediately
        start_write_index = 0
        if title_line_index == 0 and is_chapter_heading(chunk_lines[0]) and len(chunk_lines) > 1 and is_story_title(chunk_lines[1]):
             start_write_index = 1 # Start writing from the story title

        outf.writelines(chunk_lines[start_write_index:])
    chunk_count += 1

# Analysis of chunk lengths
chunk_lengths = []
if chunk_count > 0:
    for fname in os.listdir(CHUNKS_DIR):
        if fname.endswith('.md'):
            try:
                with open(os.path.join(CHUNKS_DIR, fname), encoding='utf-8') as f:
                     chunk_lengths.append(len(f.readlines()))
            except Exception as e:
                print(f"Error reading {fname}: {e}")

print(f"Chunking complete. Created {chunk_count} refined chunks.")
if chunk_lengths:
    print(f"Min/Max/Avg chunk line count: {min(chunk_lengths)}, {max(chunk_lengths)}, {sum(chunk_lengths)//len(chunk_lengths)}")
else:
    print("No chunks were created or analysis failed.")
