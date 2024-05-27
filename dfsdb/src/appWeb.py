import sys, os
from collections import Counter

# Get the parent directory
# parent_dir = os.path.dirname(os.path.realpath(__file__))[:-4]

# Add the parent directory to sys.path
# print(parent_dir)
# print(os.curdir)
# sys.path.append("D:\\UM\\year 3 sem 1+2\\FYP\\code\\DeepLearningforStudyingDrawingBehavior\\dfsdb\\model_loaders")

# print(*sys.path, sep='\n')

# Import the module from the parent directory
from model_loaders import house_loader, tree_loader, person_loader

class Results:
    def __init__(self):
        self.res = [0, 0, 0]
        self.tags=[
            [["Stress", "Anxiety"],
                          ["Low self-esteem", "Withdrawal", "Introversion"],
                          ["High self-esteem", "Fantasizing", "Extroversion"]],
                   [["Depression", "Low Energy"],
                          ["Introversion", "Low ego-strength"],
                          ["Extroversion", "Ambition", "High ego-strength"]],
                   [["Depression", "Low Energy"],
                          ["Withdrawal", "Lack of motivation", "Boredom"],
                          ["Anxiety", "Obsession"]]
                   ]
        self.descriptions = [
            [
            "Your house drawing might have features such as excessive smoke, shading of the roof, and high detailing."
            "These characteristics suggest signs of stress or anxiety, reflecting a heightened state of worry or pressure."
            "An overactive mind and feelings of being overwhelmed might be indicated through these elements in your drawing.",
            #####################################################################################################
            "Your house drawing might exhibit features such as the absence of a door, absence of windows, a door situated"
            "much above the baseline, or a missing chimney. These elements suggest possible signs of low self-esteem, introversion,"
            "or withdrawal. Such characteristics in your drawing may reflect feelings of isolation, difficulty in expressing emotions,"
            "or a tendency to retreat inward.",
            #####################################################################################################
            "Your house drawing might display features such as a very large door, very large roof, very large windows, or an open door. "
            "These elements indicate possible signs of high self-esteem, extroversion, or a tendency to fantasize. Such characteristics "
            "in your drawing may reflect confidence, openness to social interactions, or a vivid imagination."
            ] ,
            [
            "Your tree drawing might exhibit features such as a lack of leaves, a lack of branches, or a lack of roots. "
            "These elements suggest possible signs of depression or low energy, reflecting a sense of emptiness or a lack of vitality. "
            "Feelings of sadness, fatigue, or a lack of motivation might be indicated through these elements in your drawing.",
            #####################################################################################################
            "Your tree drawing might display features such as short or no branches, and a thin and small trunk. "
            "These elements indicate possible signs of introversion and low ego strength. Such characteristics in your "
            "drawing may reflect a reserved nature and a lack of self-confidence or inner strength.",
            #####################################################################################################
            "Your tree drawing might have features such as a large trunk, large branches, or a large number of leaves. "
            "These elements suggest possible signs of extroversion, ambition, or high ego strength. Such characteristics in your "
            "drawing may reflect sociability, a drive to achieve goals, or a strong sense of self-worth."
            ],
            [
            "Your person drawing might exhibit features such as a lack of details, or miniature in size. "
            "These elements suggest possible signs of depression or low energy, reflecting a sense of emptiness or a lack of vitality. "
            "Feelings of sadness, fatigue, or a lack of motivation might be indicated through these elements in your drawing.",
            #####################################################################################################
            "Your person drawing might display features such as being overly simplistic, lacks significant detail, or resembles a stick figure. "
            "These elements indicate possible signs of withdrawal, lack of motivation, or boredom. Such characteristics in your "
            "drawing may reflect a tendency to isolate oneself, a lack of interest in activities, or a sense of monotony.",
            #####################################################################################################
            "Your person drawing might have features such as high details, well-defined facial features or proportional dimensions of limbs and body. "
            "These elements suggest possible signs of anxiety, obsession, or a tendency to overthink. Such characteristics in your "
            "drawing may reflect a heightened state of worry, a preoccupation with certain thoughts, or a tendency to ruminate."
            ]
        ]
        
    def overall_result(self):
        # Get the results for each drawing type
        house_result, house_description = self.tags[0][self.res[0]], self.descriptions[0][self.res[0]]
        tree_result, tree_description = self.tags[1][self.res[1]], self.descriptions[1][self.res[1]]
        person_result, person_description = self.tags[2][self.res[2]], self.descriptions[2][self.res[2]]

        # Aggregate the results
        results_cnt = {}
        for result in [house_result, tree_result, person_result]:
            for r in result:
                print("r", r)
                results_cnt[r] = results_cnt.get(r, 0) + 1

        # Determine the most common outcome
        overall_result = max(results_cnt, key=results_cnt.get)

        # Debug
        print("overall result", overall_result)
        
        return overall_result
    
    def get_house_result(self):
        
        housePredict = [0, 0, 0]
        # 3 Predictions for the house image that was drawn
        housePredict[0] = house_loader.predict('D:\\UM\\year 3 sem 1+2\\FYP\\code\\DeepLearningforStudyingDrawingBehavior\\dfsdb\\model\\house\\house_model_10.tar', 'D:\\UM\\year 3 sem 1+2\\FYP\\code\\DeepLearningforStudyingDrawingBehavior\\dfsdb\\toPredict\\predictHouse.png')
        housePredict[1] = house_loader.predict('D:\\UM\\year 3 sem 1+2\\FYP\\code\\DeepLearningforStudyingDrawingBehavior\\dfsdb\\model\\house\\house_model_12.tar', 'D:\\UM\\year 3 sem 1+2\\FYP\\code\\DeepLearningforStudyingDrawingBehavior\\dfsdb\\toPredict\\predictHouse.png')
        housePredict[2] = house_loader.predict('D:\\UM\\year 3 sem 1+2\\FYP\\code\\DeepLearningforStudyingDrawingBehavior\\dfsdb\\model\\house\\house_model_15.tar', 'D:\\UM\\year 3 sem 1+2\\FYP\\code\\DeepLearningforStudyingDrawingBehavior\\dfsdb\\toPredict\\predictHouse.png')

        # Find the most commonly predicted label for each of the images and store it
        self.res[0] = Counter(housePredict).most_common()[0][0]
        
        print("housePredict =", housePredict)
        print("self.res =", self.res)
        return self.tags[0][self.res[0]], self.descriptions[0][self.res[0]]
    
    def get_tree_result(self):

        treePredict = [0, 0, 0]
        # 3 Predictions for the tree image that was drawn
        treePredict[0] = tree_loader.predict('D:\\UM\\year 3 sem 1+2\\FYP\\code\\DeepLearningforStudyingDrawingBehavior\\dfsdb\\model\\tree\\tree_model_10.tar', 'D:\\UM\\year 3 sem 1+2\\FYP\\code\\DeepLearningforStudyingDrawingBehavior\\dfsdb\\toPredict\\predictTree.png')
        treePredict[1] = tree_loader.predict('D:\\UM\\year 3 sem 1+2\\FYP\\code\\DeepLearningforStudyingDrawingBehavior\\dfsdb\\model\\tree\\tree_model_12.tar', 'D:\\UM\\year 3 sem 1+2\\FYP\\code\\DeepLearningforStudyingDrawingBehavior\\dfsdb\\toPredict\\predictTree.png')
        treePredict[2] = tree_loader.predict('D:\\UM\\year 3 sem 1+2\\FYP\\code\\DeepLearningforStudyingDrawingBehavior\\dfsdb\\model\\tree\\tree_model_15.tar', 'D:\\UM\\year 3 sem 1+2\\FYP\\code\\DeepLearningforStudyingDrawingBehavior\\dfsdb\\toPredict\\predictTree.png')
        
        # Find the most commonly predicted label for each of the images and store it
        self.res[1] = Counter(treePredict).most_common()[0][0]
        
        print("treePredict =", treePredict)
        print("self.res =", self.res)
        return self.tags[1][self.res[1]], self.descriptions[1][self.res[1]]
    
    def get_person_result(self):

        personPredict = [0, 0, 0]
        # 3 Predictions for the person image that was drawn
        personPredict[0] = person_loader.predict('D:\\UM\\year 3 sem 1+2\\FYP\\code\\DeepLearningforStudyingDrawingBehavior\\dfsdb\\model\\person\\person_model_10.tar', 'D:\\UM\\year 3 sem 1+2\\FYP\\code\\DeepLearningforStudyingDrawingBehavior\\dfsdb\\toPredict\\predictPerson.png')
        personPredict[1] = person_loader.predict('D:\\UM\\year 3 sem 1+2\\FYP\\code\\DeepLearningforStudyingDrawingBehavior\\dfsdb\\model\\person\\person_model_12.tar', 'D:\\UM\\year 3 sem 1+2\\FYP\\code\\DeepLearningforStudyingDrawingBehavior\\dfsdb\\toPredict\\predictPerson.png')
        personPredict[2] = person_loader.predict('D:\\UM\\year 3 sem 1+2\\FYP\\code\\DeepLearningforStudyingDrawingBehavior\\dfsdb\\model\\person\\person_model_15.tar', 'D:\\UM\\year 3 sem 1+2\\FYP\\code\\DeepLearningforStudyingDrawingBehavior\\dfsdb\\toPredict\\predictPerson.png')
        
        # Find the most commonly predicted label for each of the images and store it
        self.res[2] = Counter(personPredict).most_common()[0][0]
        
        print("personPredict =", personPredict)
        print("self.res =", self.res)
        return self.tags[2][self.res[2]], self.descriptions[2][self.res[2]]