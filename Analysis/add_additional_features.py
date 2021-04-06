## Allows adding more features from conceptnet to the feature file.
import pandas as pd

import requests
import csv

from Analysis.data_import import StudyInfo
additional_features = ["figure_lightsource","figure_container"]



class Entity:
    headings = ['Object', 'CN_ISA_CONTAINER', 'CN_UsedFor_Light']

    def __init__(self, class_name, container_value, lightsource_value):
        self.class_name = class_name
        self.container_value = container_value
        self.lightsource_value = lightsource_value
        self.csv_row = [self.class_name, self.container_value, self.lightsource_value]


def clean_name(object_name):
    if '(' in object_name:
        clean_name = object_name[:object_name.find("(")]
    elif '_' in object_name:
        clean_name = object_name[:object_name.find("_")]
    elif '.' in object_name:
        clean_name = object_name[:object_name.find(".")]
    else:
        clean_name = object_name
    return clean_name.lower().strip()


def extract_relation_weight(relation, obj1, obj2):
    object1_name = clean_name(obj1)
    object2_name = clean_name(obj2)
    tag = '/r/' + relation

    conceptlinks = requests.get(
        'http://api.conceptnet.io/query?node=/c/en/' + object1_name + '&other=/c/en/' + object2_name).json()
    if len(conceptlinks['edges']) == 0:
        return 0
    elif not any(edges['rel']['@id'] == tag for edges in conceptlinks['edges']):
        return 0
    else:
        for edges in conceptlinks['edges']:
            if edges['rel']['@id'] == tag:
                return edges['weight']

def get_properties(study_info: StudyInfo):
    with open(study_info.input_feature_data_folder + "/object properties.csv", "w") as csvfile:
        outputwriter = csv.writer(csvfile)
        outputwriter.writerow(Entity.headings)

        out = []
        with open(study_info.scene_info_csv_file, "r") as csvfile:
            reader = csv.reader(csvfile)
            datalist = list(reader)
            objects_line = datalist[2]

            for obj in objects_line:
                class_name = clean_name(obj)
                try:

                    cont = min(extract_relation_weight('IsA', class_name, 'container'), 1)
                    light = min(extract_relation_weight('UsedFor', class_name, 'light'), 1)
                    # Covering is annoying to extract from CN
                    # For example, `lid' is not a direct type of covering, but it IS A 'portal covering' which is itself a type of covering.
                    # types_of_covering = ['body', 'cloth', 'portal', 'protective', 'floor', 'decorative', 'interposed',
                    #                      'enveloping']
                    #
                    # for c in types_of_covering:
                    #     cover = extract_relation_weight('IsA', class_name, c + ' covering')
                    #     if cover >= 1:
                    #         break

                    e = Entity(class_name, cont, light)
                    out.append(e)

                    outputwriter.writerow(e.csv_row)
                except:
                    print('Commonsense property error')
                    print(clean_name)
    return out


def get_feature_value(entity_list, obj, feature):
    class_name = clean_name(obj)
    for ent in entity_list:
        if ent.class_name == class_name:
            if feature == "figure_lightsource":

                return ent.lightsource_value
            if feature == "figure_container":

                return ent.container_value


def add_properties_to_relation_file(study_info):
    property_list = get_properties(study_info)

    relation_dataset = pd.read_csv(study_info.input_feature_csv)
    # print(relation_dataset)
    lightsource_list = [get_feature_value(property_list, obj, "figure_lightsource") for obj in
                        relation_dataset['Figure'].values]
    container_list = [get_feature_value(property_list, obj, "figure_container") for obj in
                      relation_dataset['Figure'].values]

    relation_dataset["figure_lightsource"] = lightsource_list
    relation_dataset["figure_container"] = container_list
    # print(relation_dataset)
    relation_dataset.to_csv(study_info.input_feature_csv, index=False)


if __name__ == '__main__':
    s_info = StudyInfo("2019 study")
    add_properties_to_relation_file(s_info)
