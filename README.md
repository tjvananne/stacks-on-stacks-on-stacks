# stacks-on-stacks-on-stacks


## rationale/goal of this repo


The goal of this project is to create an extremely well-documented and robust framework for a multi-level 
stacked ensemble of predictive machine learning models. I am modeling this heavily based on [Keiku's 2nd 
place github repo for the Airbnb Kaggle 
competition](https://github.com/Keiku/kaggle-airbnb-recruiting-new-user-bookings) which I think is one of 
the best examples of a complex stacked ensemble on github. 


## build small - scale fast

At first, I reran Keiku's experiments exactly how he had set them up. This was really beneficial to helping me understand what he was doing in each portion of his script and why. The issue with this method was how long it takes to run each little piece of the script. There are 19 models in stage 1, each of which takes me up to 2 hours to run. This is not a conducive environment for learning, tinkering, and gaining a solid understanding of the concepts of stacking.


Therefore, I am rebuilding much of the same logic as what Keiku used, but on a **significantly smaller dataset** (the iris dataset which comes preloaded with R. Not all of these concepts will work extremely well with such a small dataset, but I'm more focused on setting up the architecture and framework of the project so I can quickly and easily apply it to much larger projects.


Ideally, I would be able to ssh into a remote virtual box, scp the experiment dir up to the remove server, edit some config variables, and just hit "go" and leave it alone. I want everything to be modular, almost like how you would train a keras neural network, but for a stacking architecture.


## Getting started


If you want to contribute to the repo or just get started understanding it better, then navigate into the "script" directory. The readme in there will give much better details on how the actual flow of the program works. The goal is to keep all of the code in this repo squeaky clean and very well documented to really lower the barriers on picking up the logic and applying it towards a different dataset.



