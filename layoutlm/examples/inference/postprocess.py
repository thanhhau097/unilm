import re
from .datasets import sort_funsd_reading_order

def empty_sub_string(st: str, subs=[], lower=False):
    if not subs:
        return st
    if lower:
        st = st.lower()
    for sub in subs:
        st = st.replace(sub, "")
    st = st.strip()
    return st


def post_process_delivery_date(text):
    if len(text) >= 3:
        if "日" in text:
            index_of_day = text.index("日")
            text = text[:index_of_day]
        text = text.strip().replace(".", "/")
        char_list = ["年", "月", "日"]
        text = re.sub("|".join(char_list), "/", text)
        text = re.sub(r"([^0-9/]+?)", "", text)

    counter = text.count("/")
    if counter > 2:
        first_index = text.index("/")
        text = text[first_index + 1:]

        second_index = text.index("/")
        text = text[second_index + 1:]
    elif 1 < counter <= 2:
        first_index = text.index("/")
        text = text[first_index + 1:]
    return text


def post_process_tel(text):
    text = text.replace('O', '0')
    text = re.sub(r"([^0-9-]+?)", "", text)
    return text


def count_digits(text):
    text = re.sub(r"([^0-9]+?)", "", text)
    return len(text)


def post_process_merge_fields(regions, formal_key):
    new_regions = []
    filtered_regions = []
    for region in regions:
        if region['type'] == formal_key:
            filtered_regions.append(region)
        else:
            new_regions.append(region)

    if len(filtered_regions) == 0:
        return regions

    filtered_regions = sort_funsd_reading_order(filtered_regions)
    combined_text = ''.join([r['text'] for r in filtered_regions])
    list_x, list_y = [], []
    for box in [r['box'] for r in filtered_regions]:
        x1, y1, x2, y2 = box
        list_x.extend((x1, x2))
        list_y.extend((y1, y2))
    x1, y1, x2, y2 = min(list_x), min(list_y), max(list_x), max(list_y)
    new_regions.append({
        'location': [[x1, y1], [x1, y2], [x2, y2], [x2, y1]],
        'text': combined_text,
        'key_type': 'value',
        'type': formal_key,
        'confidence': 0.5,
        'box': [x1, y1, x2, y2]
    })
    return new_regions


def convert_kv_format(entities, key_list=None):
    report_dict = []
    for entity in entities:
        x1, y1, x2, y2 = entity['box']
        split_index = entity['tag'].find('_')
        key_type, formal_key = entity['tag'][:split_index].lower(), entity['tag'][split_index + 1:].lower()
        if key_type != 'value':
            continue
        text = entity['text'].replace(' ', '').replace('#', '').replace('[UNK]', '').upper()
        text = post_process_fields(formal_key, text)
        if text is None:
            continue
        if key_list is not None and formal_key not in key_list:
            continue
        report_dict.append({
            'location': [[x1, y1], [x1, y2], [x2, y2], [x2, y1]],
            'text': text,
            'key_type': 'value',
            'type': formal_key,
            'confidence': 0.5,
            'box': entity['box']
        })

    regions = post_process_merge_fields(report_dict, 'company_name')
    regions = post_process_merge_fields(regions, 'delivery_destination_company_name')

    return regions


def post_process_fields(formal_key, text):
    if formal_key == 'delivery_date':
        text = post_process_delivery_date(text)
        print('date', text)

    if formal_key in ["company_tel", "delivery_destination_company_tel",
                      "company_fax", "delivery_destination_company_fax"]:
        text = post_process_tel(text)
        print('tel', text)
        if count_digits(text) < 7:
            text = None

    return text