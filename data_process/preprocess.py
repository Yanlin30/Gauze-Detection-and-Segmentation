def json_to_mask(json_path, mask_path):
    with open(json_path, encoding="UTF-8") as f:
        content = json.load(f)

    mask_width = content["imageWidth"]
    mask_height = content["imageHeight"]

    shapes = content["shapes"]
    o_points = []
    label_points = []
    for shape in shapes:
            category = shape['label']
            points = shape['points']
            if "rectangle" in category:
                continue
            elif category == "0":
                o_points.append(points)
            else:
                label_points.append(points)

    mask = np.zeros([mask_height, mask_width], np.uint8)

    for i in range(len(label_points)):
        points_array = np.array(label_points[i], dtype=np.int32)
        mask = cv2.fillPoly(mask, [points_array], 255)

    if o_points != []:
        for i in range(len(o_points)):
            points_array = np.array(o_points[i], dtype=np.int32)
            mask = cv2.fillPoly(mask, [points_array], 0)
    cv2.imwrite(mask_path, mask)
    print(mask_path)
