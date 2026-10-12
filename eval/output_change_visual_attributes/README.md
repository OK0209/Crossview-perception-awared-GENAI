# Output Change Visual Attributes

This analysis measures interpretable output-change proxies from sweep-generated images.
It does not estimate perception scores directly.

Metrics:
- enclosure_proxy = building_ratio / (sky_ratio + road_ratio + epsilon). Higher usually means stronger enclosure.
- vegetation_ratio = vegetation pixels / total pixels. Higher means more greenery.
- edge_clutter_density = Canny edge pixels / total pixels. Higher means denser edges, which can indicate detail or clutter.
- line_orientation_consistency = weighted circular resultant length of line orientations. Higher means more concentrated line directions.
- texture_clutter = std(Laplacian(gray)) / 255. Higher means stronger high-frequency texture/local variation.

Interpretation should be dimension-specific and empirical. Do not assume high is always good.

Example hypotheses:
- beautiful high may increase vegetation_ratio, reduce or moderate edge_clutter_density, or improve line_orientation_consistency.
- safety high may improve line_orientation_consistency and reduce edge_clutter_density.
- lively high may increase vegetation_ratio, edge_clutter_density, or texture_clutter.
- boring high may reduce texture_clutter, edge_clutter_density, or vegetation_ratio.
- depressing high may reduce vegetation_ratio or change enclosure/texture.

Final conclusions should come from the statistics, not from these hypotheses.
