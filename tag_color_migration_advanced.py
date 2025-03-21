#!/usr/bin/env python3
"""
Advanced Tag Color Migration Script

This script intelligently maps existing tag colors to the closest colors in the new
predefined color palette, ensuring a smooth transition while maintaining color semantics.
It includes color distance calculation to find the best match for each existing color.
"""

import sqlite3
import sys
import math
import re
import argparse

# Define the new predefined color palette
PREDEFINED_COLORS = [
    # Primary colors
    "#5787eb",  # Primary blue
    "#4fccc4",  # Teal
    "#fd565c",  # Red
    "#ffc107",  # Yellow/Warning
    
    # Complementary colors
    "#6c5ce7",  # Purple
    "#00b894",  # Green
    "#ff7675",  # Light red
    "#fdcb6e",  # Light yellow
    
    # More options
    "#e84393",  # Pink
    "#74b9ff",  # Light blue
    "#a29bfe",  # Lavender
    "#55efc4",  # Mint
]

def hex_to_rgb(hex_color):
    """Convert a hex color string to RGB values."""
    # Remove the # if present
    hex_color = hex_color.lstrip('#')
    
    # Handle both 3-digit and 6-digit hex
    if len(hex_color) == 3:
        r = int(hex_color[0] + hex_color[0], 16)
        g = int(hex_color[1] + hex_color[1], 16)
        b = int(hex_color[2] + hex_color[2], 16)
    elif len(hex_color) == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    else:
        # Default to black for invalid hex
        return (0, 0, 0)
    
    return (r, g, b)

def color_distance(color1, color2):
    """Calculate the Euclidean distance between two colors in RGB space."""
    r1, g1, b1 = hex_to_rgb(color1)
    r2, g2, b2 = hex_to_rgb(color2)
    
    # Euclidean distance formula for 3D space (RGB)
    return math.sqrt((r2 - r1) ** 2 + (g2 - g1) ** 2 + (b2 - b1) ** 2)

def is_valid_hex_color(color):
    """Check if a string is a valid hex color."""
    if not color:
        return False
    
    # Match both 3-digit and 6-digit hex colors with or without #
    pattern = r'^#?([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$'
    return re.match(pattern, color) is not None

def find_closest_color(target_color, color_palette):
    """Find the closest color in the palette to the target color."""
    if not is_valid_hex_color(target_color):
        # If the current color is invalid, return the default primary color
        return color_palette[0]
    
    # Ensure the target color has a # prefix
    if not target_color.startswith('#'):
        target_color = '#' + target_color
    
    # Calculate distance to each color in the palette
    distances = [(color, color_distance(target_color, color)) for color in color_palette]
    
    # Find the color with the smallest distance
    closest_color = min(distances, key=lambda x: x[1])[0]
    
    return closest_color

def connect_to_database(db_path):
    """Connect to the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

def get_all_tags(conn):
    """Retrieve all tags from the database."""
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id, name, color FROM tags")
        return cursor.fetchall()
    except sqlite3.Error as e:
        print(f"Error retrieving tags: {e}")
        return []

def update_tag_color(conn, tag_id, new_color):
    """Update the color of a specific tag."""
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE tags SET color = ? WHERE id = ?", (new_color, tag_id))
        conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"Error updating tag {tag_id}: {e}")
        conn.rollback()
        return False

def migrate_tag_colors(db_path, dry_run=True):
    """Migrate all tag colors to the new predefined palette."""
    conn = connect_to_database(db_path)
    tags = get_all_tags(conn)
    
    if not tags:
        print("No tags found in the database.")
        conn.close()
        return
    
    print(f"Found {len(tags)} tags in the database.")
    print("-" * 60)
    
    updated_count = 0
    skipped_count = 0
    error_count = 0
    
    # Track which predefined colors are already being used
    # This helps distribute colors more evenly if we're adding a lot of new tags
    color_usage_count = {color: 0 for color in PREDEFINED_COLORS}
    
    for tag in tags:
        tag_id = tag['id']
        tag_name = tag['name']
        current_color = tag['color']
        
        # Skip tags that already have one of our predefined colors
        if current_color in PREDEFINED_COLORS:
            print(f"✓ Tag '{tag_name}' already uses a predefined color: {current_color}")
            color_usage_count[current_color] += 1
            skipped_count += 1
            continue
        
        # Find the closest color from our palette
        new_color = find_closest_color(current_color, PREDEFINED_COLORS)
        color_usage_count[new_color] += 1
        
        print(f"→ Tag '{tag_name}': changing color from {current_color} to {new_color}")
        
        # Update the tag color in the database (unless this is a dry run)
        if not dry_run:
            success = update_tag_color(conn, tag_id, new_color)
            if success:
                updated_count += 1
            else:
                error_count += 1
        else:
            updated_count += 1
    
    print("\n" + "-" * 60)
    print("Summary:")
    print(f"- Tags found: {len(tags)}")
    print(f"- Tags to update: {updated_count}")
    print(f"- Tags already using predefined colors: {skipped_count}")
    
    if not dry_run:
        print(f"- Errors during update: {error_count}")
    
    # Show color distribution
    print("\nColor distribution after migration:")
    for color, count in color_usage_count.items():
        print(f"  {color}: {count} tags")
    
    if dry_run:
        print("\nThis was a DRY RUN. No changes were made to the database.")
        print("To actually update the colors, run the script with --apply")
    else:
        print("\nTag colors have been updated successfully!")
    
    conn.close()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Migrate tag colors to the new predefined palette.')
    parser.add_argument('db_path', help='Path to the SQLite database file')
    parser.add_argument('--apply', action='store_true', help='Actually apply the changes (otherwise dry run)')
    args = parser.parse_args()
    
    # Run the migration
    migrate_tag_colors(args.db_path, dry_run=not args.apply)
