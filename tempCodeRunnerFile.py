def assign_tiles_to_corners():
    global fixed_corners
    global tiles
    
    num_corners = len(fixed_corners)
    num_tiles = int(num_corners / 2) - 1 
    # tiles = {}
    
    for i in range(num_tiles):
        tile_corners = fixed_corners[i*2:i*2+4]  # Each tile has 4 corners
        tiles[f"Tile {i+1}"] = tile_corners
    
    return tiles