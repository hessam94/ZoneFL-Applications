# Zone-FL
This project implements ZFL, a zone-based [Federated Learning](https://federated.withgoogle.com/) architecture that builds and manages different models for different geographical zones to mitigate the problem of non-IID data. The main idea of ZFL is to adapt each zone model to the data and behaviors of the users who frequently spend time in that zone. ZFL is designed to work in a mobile-edge-cloud infrastructure, where FL training in each zone is managed by an edge node. The cloud plays a coordinator role to support zone management, load-balancing, and fault tolerance across edge nodes. We implemented and evaluated two proof-of-concept models based on ZFL for two different types of ML problems, a vehicular traffic prediction model and a mobile user activity recommendation model.

These projects have been implemented offline and we don't have different client-server machines here. This means we run the programs for each group of data(location Zones) separately and then compare it with traditional FL. The results show the zone based models built with ZFL lead to higher accuracy compared to global FL models. 

ZoneFL had some advantage compare to central FL. Current FL solutions, do not address well the requirements for good model accuracy and adaptability to user mobility behavior. FL training for large numbers of mobile users spread over large areas suffers from lower accuracy due to the non-Independent and Identically Distributed (non-IID) data distribution, specific to mobile users whose behaviors are location dependent. The main idea of ZFL is to divide the space into a number of
zones with similar user behaviors and train FL models for each of these zones. As users move from one zone to another, their mobile devices also switch from one zone model to another. In this way, ZFL adapts each zone model to the data and behaviors of the users who frequently spend time in that zone. Many mobile apps can benefit from ZFL models trained on mobile sensing data. 

![ZoneFL Overview](/Zone-Fl-Overview.png)
*Zone-FL Overview*  

# implementation
To evaluate the benefits of ZFL we developed two models based on our system architecture: a Vehicular Traffic Prediction model that predicts the speed on road segments in each zone, and an User Activity Recommendation model that recommends activities in a city. Since our main objective in this work is to show the advantage of ZFL over Centralized FL (CFL) in terms of accuracy, we use simple existing models and adapt them to work under the ZFL architecture. For ZFL, we train and test each model independently for each zone, while for CFL, we train and test the same model for the entire space (i.e., all the zones). The models are implemented with [TensorFlow](https://www.tensorflow.org/), [Keras](https://keras.io/), and [DL4J](https://deeplearning4j.konduit.ai/).
### 1-Vehicular Traffic Prediction
This model predicts the speed on road segments based on the time of the day and several road segment features. Data format. The input data is represented as follows:
X=(segmentType, segmentSpeedLimit, segmentLaneCount,segmentLength, time). The output is Y=(segmentSpeed). Each data record reflects the information of one car and the zone that the car traveled in. The zone is determined based on the segment ID, which reflects its location. If a segment is located in two different zones, we divide it into two different subsegments. Thus, each data point reflects only one zone.
#### dataset
The dataset for this model was produced by [SUMO](https://ieeexplore.ieee.org/abstract/document/6468040), a vehicular traffic simulator, using real-life traffic data from Cologne, Germany  

### 2- User Activity Recommendation
To test ZFL on a different type of problem, we developed a recommendation system that can predict the next user activity based on a prior sequence of activities. Data format. The input of the model is X = (x1; x2; :::; xn), a time-series of activity ids, and the output is Y = (xn+1), the recommended activity. Each activity id reflects one activity category (e.g., food=1, communication=2, outdoor=3, etc.). We tested different lengths of the activity sequence, and the best results were achieved for length=8. Furthermore, in each sequence, all the activities happened in one day. As the users are moving across the city, activities
may be associated with different zones. In this case, we split the sequences per zone.
#### dataset
We used the [Gowalla](https://dl.acm.org/doi/10.1145/2661829.2662002) public data set for the activity recommendation program. Gowalla is a locationbased social network, in which people’s location is stored under an activity category

# Supervisor
this project has been done under supervision of [Dr. Cristian Borcea](https://web.njit.edu/~borcea/)
