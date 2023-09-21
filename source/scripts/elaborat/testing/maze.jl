using Images

function get_black_pixel_coordinates(image_path)
    
    img_org = load(image_path)
    img = Gray.(img_org)
    
    height, width = size(img)

    black_pixel_coordinates = []

    for y in 1:height
        for x in 1:width
            pixel_value = img[y, x]

            if pixel_value == Gray(0.0)

                push!(black_pixel_coordinates, (x, y))
            end
        end
    end

    return black_pixel_coordinates
end

# Example usage
image_path = "elaborat/zkouska.png"
coordinates = get_black_pixel_coordinates(image_path)

# Display the coordinates of black pixels

#=
println("Coordinates of black pixels:")
for coord in coordinates
    println(coord)
end

=#
