# Code in this file is slightly adapted from:
# https://github.com/jsvine/pdfplumber/issues/242#issuecomment-668448246
import pdfplumber


def curves_to_edges(cs):
    """See https://github.com/jsvine/pdfplumber/issues/127"""
    edges = []
    for c in cs:
        edges += pdfplumber.utils.rect_to_edges(c)
    return edges


# Table settings.
def table_settings(page):
    return {
        "vertical_strategy": "explicit",
        "horizontal_strategy": "explicit",
        "explicit_vertical_lines": curves_to_edges(page.curves + page.edges),
        "explicit_horizontal_lines": curves_to_edges(page.curves + page.edges),
        "intersection_y_tolerance": 10,
    }


def get_bounding_boxes(page):
    return [table.bbox for table in page.find_tables(table_settings=table_settings(page))]


def table_filter(bboxes, obj):
    """Check if the object is in any of the table's bbox."""
    def obj_in_bbox(_bbox):
        """See https://github.com/jsvine/pdfplumber/blob/stable/pdfplumber/table.py#L404"""
        v_mid = (obj["top"] + obj["bottom"]) / 2
        h_mid = (obj["x0"] + obj["x1"]) / 2
        x0, top, x1, bottom = _bbox
        return (h_mid >= x0) and (h_mid < x1) and (v_mid >= top) and (v_mid < bottom)
    return not any(obj_in_bbox(__bbox) for __bbox in bboxes)
