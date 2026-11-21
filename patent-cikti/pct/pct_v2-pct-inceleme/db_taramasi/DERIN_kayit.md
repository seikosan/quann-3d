# Derin inceleme — 1. istem metinleri (uretim: derin.py, 2026-11-19)


## US20250156898A1
- neden: istem 1(a)+(e): ozellik nokta bulutu + aciklanabilirlik iddiasi
- sayfa: 877365 karakter · ham: `patentlar/US20250156898A1.md`
- OZET: A system and method for AI-driven advertising and content generation that integrates connectionist and symbolic AI techniques to deliver personalized, contextually relevant user experiences. The invention leverages specialized agent networks, knowledge graphs, and retrieval-augmented generation to create dynamic, adaptive content with seamlessly integrated advertisements. It employs a sophisticated ad integration layer and experience broker to ensure relevance and engagement. The system prioritizes security, traceability, and user preferences while optimizing ad placement and content delivery. By bridging connectionist and symbolic AI, the platform enables immersive, tailored interactions ac
- 1. ISTEM: What is claimed is:
1. A system for AI based tailored advertising content generation, comprising:
a computing device comprising at least a memory and a processor;
a plurality of programming instructions that, when operating on the processor, cause the computing device to:
receive a user query;
process the query through a plurality of specialized agents;
facilitate agent-to-agent interactions between the plurality of specialized agents, wherein the plurality of specialized agents transmit advertising data between themselves and to either a knowledge graph, a retrieval-augmented generation system, or both;
access either the knowledge graph, the retrieval-augmented generation system, or both using a distributed computational graph which may utilize the information within the knowledge graph or retrieval-augmented generation system to retrieve information and generate a query response that includes curated advertising content;
optimize the layout and appearance of the curated advertising content within the query response;
broadcast the query response to a user through a user interface;
collect user feedback based on how the user responds to the query response; and
continuously update curated advertising content based on the user feedback.
2. The system of claim 1, wherein the plurality of specialized agents includes at least a marketplace agent and a product agent.
3. The system of claim 1, wherein the system further comprises an experience broker configured to manage the agent-to-agent interactions and orchestrate data exchange between the specialized agents.
4. The system of claim 1, wherein the distributed computational graph is configured to select and execute different generative AI models based on the nature of the query and the type of response required.
5. A method for AI based tailored advertising content generation, comprising the steps of:
receiving a user query;
processing the query through a plurality of specialized agents;
facilitate agent-to-agent interactions between the plurality of specialized agents, wherein the plurality of specialized agents transmit advertising data between themselves and to either a knowledge graph, a retrieval-augmented generation system, or both;
accessing either the knowledge graph, the retrieval-augmented generation system, or both using a distributed computational graph which may utilize the information within the kno


## US12039636B2
- neden: istem 1(c): deep set tabanli semetrik agregasyon
- sayfa: 138846 karakter · ham: `patentlar/US12039636B2.md`
- OZET: For reconstruction in medical imaging using a scan protocol with repetition, a machine learning model is trained for reconstruction of an image for each repetition. Rather than using a loss for that repetition in training, the loss based on an aggregation of images reconstructed from multiple repetitions is used to train the machine learning model. This loss for reconstruction of one repetition based on aggregation of reconstructions for multiple repetitions is based on deep set-based deep learning. The resulting machine-learned model may better reconstruct an image from a given repetition and/or a combined image from multiple repetitions than a model learned from a loss per repetition.

## 
- 1. ISTEM: What is claimed is:

1. A method of machine training for magnetic resonance (MR) reconstruction in medical imaging, the method comprising:

acquiring training data for an MR protocol using multiple repetitions;

machine training a neural network for the MR reconstruction using the training data, the neural network training for the MR reconstruction to output an image for each one of the multiple repetitions, a first loss used in the training being based on an aggregation of the images from the multiple repetitions, the first loss comprising a measure of difference between the aggregation of the images and a ground truth image such that the neural network learns for each of the multiple repetitions based on the aggregation across the multiple repetitions; and

storing the neural network as machine trained.

2. The method of

claim 1
wherein machine training comprises training the neural network to output the image for each one of the multiple repetitions independently of the output for other ones of the multiple repetitions.

3. The method of

claim 1
wherein machine training comprises training the neural network to output the image for at least one of the multiple repetitions in dependence on information from the neural network used to output an image for another one of the multiple repetitions.

4. The method of

claim 3
wherein the dependence comprises a global pooling across instances of the neural network for respective ones of the multiple repetitions.

5. The method of

claim 1
wherein acquiring the training data comprises acquiring with the MR protocol being a diffusion-weighted, turbo-spin-echo, contrast with different echo times, or contrast with different flip angles protocol.

6. The method of

claim 1
wherein machine training comprises machine training the neural network for repetition reconstruction preserving invariance by permutation with respect to the multiple repetitions.

7. The method of

claim 1
wherein machine training comprises machine training where the aggregation is an average, a geometric product, or a geometric mean.

8. The method of

claim 1
wherein the neural network comprises an input layer configured to accept information for a single one of the multiple repetitions for the output of the image for that single one of the multiple repetitions.

9. The method of

claim 1
wherein the neural network comprises an input layer having


## CN117392424A
- neden: istem 1(b)+(c): geometri ozellikleri + deep sets siniflandirma
- sayfa: 95177 karakter · ham: `patentlar/CN117392424A.md`
- OZET: 本发明提供一种基于多几何双边缘注意力网络的三维点云分类方法，基于注意力机制构建多几何双边缘注意力网络。该网络具有多几何增强模块、双边缘注意力模块和通道注意力模块。注意力机制使网络更好地关注输入数据中对预测更重要的部分，从而提升三维点云分类的精度。多几何增强模块考虑欧式空间特征和几何特征，在不同方向获取点云原始特征，以获得更多几何线索。双边缘注意力模块从低层几何输入中提取更有意义信息并适当加权从而聚合点特征。该双边缘注意力模块用两个不同分支分别有效学习局部和全局特征，最后连接两分支以聚合本地和全局上下文要素。通道注意力模块在最终输出中提取有意义信息和适当加权聚合点特征进行三维点云分类任务。

The present invention provides a three-dimensional point cloud classification method based on a multi-geometry dual-edge attention network, and constructs a multi-geometry dual-edge attention network based on an attention mechanism. The network has a multi-geometry enhancement module, a dual-edge attention module, and a channel attention module. The attention mechanism allows the network to better 
- 1. ISTEM: Claims (8)

1.一种基于多几何双边缘注意力网络的三维点云分类方法，其特征在于，包括以下步骤：1. A 3D point cloud classification method based on a multi-geometric dual-edge attention network, characterized by comprising the following steps:步骤S1，基于注意力机制构建多几何双边缘注意力网络，该网络具有多几何增强模块、重复堆叠的双边缘注意力模块，以及通道注意力模块，所述注意力机制包含局部注意力和全局注意力；Step S1: Construct a multi-geometric dual-edge attention network based on the attention mechanism. This network has a multi-geometric enhancement module, repeatedly stacked dual-edge attention modules, and a channel attention module. The attention mechanism includes local attention and global attention.步骤S2，将原始点云的无序点集输入至所述多几何双边缘注意力网络中，通过所述多几何增强模块获取所述原始点云在不同方向的点云特征，从而生成多个维度低级几何向量；Step S2: Input the unordered point set of the original point cloud into the multi-geometric dual-edge attention network, and obtain the point cloud features of the original point cloud in different directions through the multi-geometric enhancement module, thereby generating multi-dimensional low-level geometric vectors.步骤S3，所述双边缘注意力模块具有边缘特征分支和密度边缘注意力分支，这两个分支以所述多个维度低级几何向量为输入，基于所述局部注意力分别提取所述原始点云的全局点特征和局部特征，并将提取到的特征以跳跃连接的方式连接最终得到3D点云特征向量；Step S3: The dual edge attention module has an edge feature branch and a density edge attention branch. These two branches take the multiple-dimensional low-level geometric vectors as input, extract the global point features and local features of the original point cloud based on the local attention, and connect the extracted features in a skip connection manner to finally obtain the 3D point cloud feature vector.步骤S4，所述通道注意力模块以所述多个维度低级几何向量为输入，基于所述全局注意力生成通道注意力掩码，通过该通道注意力掩码对所述3D点云特征向量进行缩放，从而进一步增强特征表示能力；Step S4: The channel attention module takes the multiple low-level geometric vectors of the multiple dimensions as input, generates a channel attention mask based on the global attention, and scales the 3D point cloud feature vector through the channel attention mask, thereby further enhancing the feature representation capability.步骤S5，所述多几何双边缘注意力网络采用最大池化和平均池化收集所述缩放后特征的具有代表性的特征向量，并通过MLP层将收集到的特征向量逐元素合并得到输出结果数值，以该数值判断所述原始点云的所属类别。Step S5: The multi-geometric dual-edge attention network uses max pooling and average pooling to collect representative feature vectors of the scaled features, and merges the collected feature vectors element by element through the MLP layer to obtain the output value, which is used to determine the category of the original point cloud.2.根据权利要求1所述的基于多几何双边缘


## WO2026046480A1
- neden: istem 1(e)(i): girdi cikarma ile AI testi/aciklama
- sayfa: 157685 karakter · ham: `patentlar/WO2026046480A1.md`
- OZET: The invention relates to a computer-implemented method for testing a function of a technical system (130) in a test case, the method having a first step (S1) of receiving, in a request input part (311), one or more text-based requests which define at least one test case, said requests at least partly comprising natural language; a second step (52) of generating at least one regular expression for each text-based request (350) in a request checking part (312); a third step (53) of generating a test-case database (350) from the regular expressions for each request in a data structuring part (340); a fourth step (S4) of receiving one or more pieces of test-case-specific parameter information in
- 1. ISTEM: 


## US12190228B2
- neden: istem 9/14: INT8 agirlik + lookup table + kenet cihaz
- sayfa: 205103 karakter · ham: `patentlar/US12190228B2.md`
- OZET: The disclosed embodiments relate to a system that generates and executes a deep neural network (DNN) based on target runtime parameters. During operation, the system receives a trained original model and a set of target runtime parameters for the DNN, wherein the target runtime parameters are associated with one or more of the following for the DNN: desired operating conditions, desired resource utilization, and desired accuracy of results. Next, the system generates a context-specific model based on the original model and the set of target runtime parameters. The system also generates an operational plan for executing both the original model and the context-specific model to meet requiremen
- 1. ISTEM: What is claimed is:

1. A method for generating and executing a deep neural network (DNN) based on target runtime parameters, comprising:

receiving a trained original model and a set of target runtime parameters for the DNN, wherein the target runtime parameters are associated with one or more of the following for the DNN: desired operating conditions, desired resource utilization, and desired accuracy of results;

generating a context-specific model based on the original model and the set of target runtime parameters;

generating an operational plan for executing both the original model and the context-specific model to meet requirements of the target runtime parameters; and

controlling execution of the original model and the context-specific model based on the operational plan; and

deploying and executing the context-specific model at a location in a hierarchy of computing nodes, wherein the location is determined based on the target runtime parameters.

2. The method of

claim 1
, wherein the target runtime parameters are generated based on current operating conditions at the location in a hierarchy of computing nodes.

3. The method of

claim 1
, wherein information regarding the target runtime parameters, locations of the original model and the context-specific model in the hierarchy of computing nodes, and results produced by the original model and the context-specific model are stored at a common location in the hierarchy of computing nodes, wherein the information is used to control execution of the operational plan.

4. The method of

claim 1
, wherein the operational plan involves executing and obtaining results from the original model when results from the context-specific model do not meet requirements of the target runtime parameters.

5. The method of

claim 4
, wherein the original model and the context-specific model are executed on different devices in the hierarchy of computing nodes.

6. The method of

claim 1
, wherein the context-specific model is moved to a location in the hierarchy of computing nodes, which is closer to a data source for the context-specific model.

7. The method of

claim 1
, wherein the context-specific model is designed to operate within constraints of computing capabilities of a specific device in the hierarchy of computing nodes.

8. The method of

claim 1
, wherein the context-specific model performs inference


## US12510888B2
- neden: istem 1(e)(ii-iii): bolge/kume tabanli aciklama + dagilim karsilastirma
- sayfa: 508081 karakter · ham: `patentlar/US12510888B2.md`
- OZET: Techniques are provided herein for creating well-balanced computer-based reasoning systems and using those to control systems. The techniques include receiving a request to determine whether to use one or more particular data elements, features, cases, etc. in a computer-based reasoning model (e.g., as data elements, cases or features are being added, or as part of pruning existing features or cases). Conviction measures are determined and inclusivity conditions are tested. The result of comparing the conviction measure can be used to determine whether to include or exclude the feature, case, etc. in the model and/or whether there are anomalies in the model. A controllable system may then be
- 1. ISTEM: What is claimed is:
1. A method comprising:
training a computer-based reasoning model, wherein the computer-based reasoning model includes a plurality of data elements, wherein each data element comprises context data paired with action data, wherein the action data is descriptive of an action taken in response to a context described by the context data;
receiving a request to determine whether one or more particular data elements of the plurality of data elements included in the computer-based reasoning model meet inclusivity conditions;
determining one or more conviction scores for the one or more particular data elements,
wherein determining the one or more conviction scores for the one or more particular data elements comprises determining an excluding-type surprisal score for the one or more particular data elements and determining a including-type surprisal score for the one or more particular data elements;
wherein:
the excluding-type surprisal score is calculated based on a first probability density or mass functions (PDMF) for a first set of data elements associated with the computer-based reasoning model where the one or more particular data elements are excluded from the first set of data elements, and
the including-type surprisal score is calculated based on a second PDMF for a second set of data elements associated with the computer-based reasoning model where the one or more particular data elements are included in the second set of data elements;
determining whether the one or more conviction scores meet one or more inclusivity conditions;
in response to determining that the one or more conviction scores meet the one or more inclusivity conditions:
including the one or more particular data elements in the computer-based reasoning model when the inclusivity conditions comprise an inclusion condition; and
excluding the one or more particular data elements in the computer-based reasoning model when the inclusivity conditions comprise an exclusion condition,
wherein determining whether the one or more conviction scores meet the inclusivity conditions comprises determining that the one or more particular data elements meet the inclusion condition when a difference between the excluding-type surprisal score and the including-type surprisal score is beyond a threshold; and
causing control of a controllable system with the computer-based reasoning mod


## WO2026170161A1
- neden: istem 1(a): sembolik cok boyutlu matris/gosterim
- sayfa: 215464 karakter · ham: `patentlar/WO2026170161A1.md`
- OZET: 
- 1. ISTEM: 


## US20260017347A1
- neden: istem 1(e)(ii): latent uzay kume etiketleme
- sayfa: 422724 karakter · ham: `patentlar/US20260017347A1.md`
- OZET: A method is provided for optimizing dynamic routing in capsule networks using temporal-spatial latent space fusion. The method includes training a temporal autoencoder to encode sequential patterns and a spatial autoencoder to capture spatial features. Input data is transformed into temporal and spatial latent representations, which are then fused using a generative adversarial network (GAN). The GAN's generator produces a unified representation and corresponding routing coefficients, which are evaluated by a discriminator based on their effect on capsule network performance. These routing coefficients are used to guide dynamic routing, enabling the network to integrate temporal and spatial 
- 1. ISTEM: What is claimed is:1.   TT1. A method for optimizing dynamic routing in capsule networks using temporal-spatial latent space fusion, comprising:

training a temporal autoencoder to process sequential data and capture latent space representations that encapsulate temporal patterns and dependencies;

training a spatial autoencoder to process static data and capture latent space representations that encapsulate spatial patterns and relationships;

transforming input data into temporal latent space representations using the temporal autoencoder;

transforming input data into spatial latent space representations using the spatial autoencoder;

fusing the temporal and spatial latent space representations using a generative adversarial network (GAN), wherein the generator combines the temporal and spatial latent spaces into a unified representation and generates routing coefficients;

evaluating the generated routing coefficients using a discriminator by assessing their impact on the performance of the capsule network;

feeding the GAN-generated routing coefficients into the capsule network to guide the dynamic routing process, enabling the network to leverage both temporal and spatial features simultaneously; and

adjusting the routing coefficients during training iterations to optimize the routing process based on the fused temporal-spatial latent space representations. 
2.   TT2. The method of claim TT1, wherein the input data comprises video data, and the temporal autoencoder captures sequential patterns from the video frames. 
3.   TT3. The method of claim TT1, wherein the spatial autoencoder captures spatial patterns from static images within the video frames. 
4.   TT4. The method of claim TT1, further comprising preprocessing the input data to enhance clarity and normalize contrast before training the temporal and spatial autoencoders. 
5.   TT5. The method of claim TT1, further comprising using a self-organizing map (SOM) to organize the fused temporal and spatial latent space representations before generating the routing coefficients with the GAN. 
6.   TT6. The method of claim TT1, wherein the temporal autoencoder uses a recurrent neural network (RNN) or long short-term memory (LSTM) architecture to capture temporal patterns and dependencies. 
7.   TT7. The method of claim TT1, wherein the spatial autoencoder uses a convolutional neural network (CNN) arc

