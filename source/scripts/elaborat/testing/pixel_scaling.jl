using Images

function get_black_pixel_coordinates_scaled(image_path, target_width, target_height)

    img_org = load(image_path)
    img = Gray.(img_org)

    height, width = size(img)

    black_pixel_coordinates = Set{Tuple{Int, Int}}()

    x_scale = target_width / width
    y_scale = target_height / height
    

    for y in 1:height
        for x in 1:width

            pixel_value = img[y, x]

            if pixel_value == Gray(0.0)

                scaled_x = round(Int, x * x_scale)
                scaled_y = round(Int, y * y_scale)

                push!(black_pixel_coordinates, (scaled_x, scaled_y))
            end
        end
    end

    return collect(black_pixel_coordinates)
end

image_path = "elaborat/zkouska.png"
target_width = 200
target_height = 200
coordinates = get_black_pixel_coordinates_scaled(image_path, target_width, target_height)

println("Scaled Coordinates of black pixels:")
for coord in coordinates
    println(coord)
end
