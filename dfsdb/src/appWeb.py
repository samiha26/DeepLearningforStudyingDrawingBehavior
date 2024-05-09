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
            "If your responsibilities are causing you stress, try organizing your tasks. Get a friend and do "
            "something you enjoy. Work on your tasks in small intervals and remember to take time for yourself!",
            #####################################################################################################
            "If you're feeling uncertain with yourself take some time off and do things that make you happy. "
            "Socializing can sometimes be taxing, dedicate some time for yourself, watch a show that you like, "
            "reach out to people that are close to you even if you're far away!",
            #####################################################################################################
            "Feeling comfortable with yourself is a great deal! Socializing is the perfect way to create "
            "memories, meet new people and gain experience. Keep working towards your goals and always take "
            "care of yourself. "
            ] ,
            [
            "Feeling down and demotivated happens to everyone, especially if you are away from people that make "
            "you feel safe and happy. It is important to allow yourself time and to not give up. A tired mind "
            "can't see clearly, allow yourself time to rest  and take things one step at a time",
            #####################################################################################################
            "Staying motivated is not an easy task, especially if you are faced with problems and challenges. "
            "Devoting time to yourself is key improve your wellbeing. Expose yourself to new  experiences and "
            "reflect on the positive aspects of your  personality.",
            #####################################################################################################
            "Having a clear goal in mind is key to helping yourself stay motivated even in the most dire "
            "situations. Enjoy time with friends, expose yourself to new experiences and keep looking ahead! "
            ],
            [
            "Feeling down and demotivated happens to everyone, especially if you are away from people that make "
            "you feel safe and happy. It is important to allow yourself time and to not give up. A tired mind "
            "can't see clearly, allow yourself time to rest and take things one step at a time.",
            #####################################################################################################
            "Withdrawal and/or lack of motivation and/or boredom Finding motivation is not always an easy task. "
            "Keeping yourself on a schedule will help you deal with your responsibilities and give you a sense "
            "of accomplishment. Expose yourself to  new experiences, discover what you enjoy the most and "
            "always dedicate for you!",
            #####################################################################################################
            "Feeling overwhelmed happens to everyone. Dealing with pressure is not always trivial or "
            "straightforward. It is important take breaks and spend time doing activities for your being. "
            "Always try to keep a balance between your obligations and leisure. "
            ]
        ]
        
    def overall_result(self):
        # Get the results for each drawing type
        house_result, _ = self.get_house_result()
        tree_result, _ = self.get_tree_result()
        person_result, _ = self.get_person_result()

        # Aggregate the results
        all_results = [house_result, tree_result, person_result]

        # Determine the most common outcome
        overall_result = Counter(all_results).most_common(1)[0][0]

        # Debug
        print("self.res =", self.res)
        
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