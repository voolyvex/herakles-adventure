import os
import re

CHUNKS_DIR = "lore_chunks"

# Function to clean up a chunk
def clean_chunk(content):
    # Split into lines
    lines = content.split('\n')
    
    # Process title - remove leading whitespace and standardize
    if lines[0].strip():
        title = lines[0].strip()
        # Remove trailing period if present
        title = title.rstrip('.')
        # Capitalize first letter if not already
        title = title[0].upper() + title[1:]
        # Remove multiple spaces
        title = ' '.join(title.split())
        lines[0] = title
    
    # Process body text
    cleaned_lines = [lines[0]]  # Start with title
    
    # Process remaining lines
    for i in range(1, len(lines)):
        line = lines[i].strip()
        if line:
            # Remove multiple spaces
            line = ' '.join(line.split())
            # Add proper indentation
            if not line.startswith('  '):
                line = '  ' + line
            cleaned_lines.append(line)
            # Add blank line after paragraph if next line is also content
            if i < len(lines) - 1 and lines[i+1].strip():
                cleaned_lines.append('')
    
    # Remove extra blank lines at end
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()
    
    return '\n'.join(cleaned_lines)

# Process all chunks
for filename in os.listdir(CHUNKS_DIR):
    if filename.endswith('.md'):
        filepath = os.path.join(CHUNKS_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            cleaned_content = clean_chunk(content)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            
            print(f"Cleaned {filename}")
            
        except Exception as e:
            print(f"Error cleaning {filename}: {str(e)}")

print("Chunk cleaning complete!")
