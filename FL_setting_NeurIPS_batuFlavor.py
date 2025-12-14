import random
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import time
class FederatedLearning:
    def __init__(self, mode, num_users, device, 
                    cos_similarity, model, TrainSetUsers, epochs, optimizer, criteron, fraction, 
                    testloader, lr, train_mode, keepProbAvail, keepProbNotAvail, bufferLimit, theta_inner):
        
        #Gradients at users
        self.grad_per_user = [[torch.zeros_like(param).to(device) for param in model.parameters()] for _ in range(num_users)]

        #Sparse gradients of users 
        self.sparse_gradient = [[torch.zeros_like(param).to(device) for param in model.parameters()] for _ in range(num_users)]


        #Weights in each user 
        self.w_user = [[param.data.clone() for param in model.parameters()] for _ in range(num_users)]

        self.w_global = [param.data.clone() for param in model.parameters()]
        
        self.mode = mode
        self.num_users = num_users
        self.device = device
        self.cos_similarity = cos_similarity
        self.model = model
        self.TrainSetUsers = TrainSetUsers 
        self.epochs = epochs
        self.optimizer = optimizer
        self.criteron = criteron
        self.fraction = fraction
        self.testloader = testloader
        self.UserAgeUL = torch.ones(self.num_users, 1).to(device)
        self.UserAgeDL = torch.ones(self.num_users, 1).to(device) 
        self.UserAgeMemory = torch.zeros(self.num_users, 1).to(device)
        self.allOnes = torch.ones(self.num_users, 1).to(device)
        self.lastAge = torch.zeros(self.num_users, 1).to(device)
        self.sum_terms = [torch.zeros_like(param).to(self.device) for param in self.w_global]
        self.lastSum_terms = [torch.zeros_like(param).to(self.device) for param in self.w_global]
        self.selected_users = None
        self.keepProbAvail = keepProbAvail
        self.keepProbNotAvail = keepProbNotAvail
        self.intermittentStateOneHot = np.array([1 if self.keepProbAvail[u] > random.random() else 0 for u in range(num_users)])
        self.intermittentUsers = np.where(self.intermittentStateOneHot)
        self.bufferSize = 0
        self.bufferLimit = bufferLimit
        self.ageAcc = 0
        self.numUp = 0
        self.lp_cos_val = 0
        self.simQueue = deque(maxlen=5)
        self.gradientMags = deque(maxlen=self.num_users)
        self.lr = lr
        self.SEst = 0
        self.userListUL = set(range(self.num_users))
        self.userListDL = set(range(self.num_users))
        self.setAllUsers = set(range(self.num_users))
        self.theta_inner = theta_inner
        self.nu_orthogonal = 5.67 #tan(80)
        self.maxAccuracy = 0
        self.patience = 0
        self.train_mode = train_mode
        self.gradientAcumDivider = 3

        self.pi = self.calculate_policy()
        self.contribution = np.zeros((self.num_users, 1))
        self.num_send = 0

        self.adamMomentum = [torch.zeros_like(param).to(self.device) for param in self.w_global]
        self.adamVariance = [torch.zeros_like(param).to(self.device) for param in self.w_global]
        self.beta1 = 0.9
        self.beta2 = 0.99
    
    def lp_cosine_similarity(self, x: torch.Tensor, y: torch.Tensor, p: int = 2) -> float:
        """
        Compute the Lp cosine similarity between two flattened gradient vectors.
    
        Args:
            x (torch.Tensor): 1D tensor.
            y (torch.Tensor): 1D tensor.
            p (int): Norm degree (e.g., 2 for L2).
    
        Returns:
            float: The Lp cosine similarity.
        """
        norm_x = torch.norm(x, p=p)
        norm_y = torch.norm(y, p=p)
        norm_x_plus_y_sq = torch.norm(x + y, p=p) ** 2
        norm_x_sq = norm_x ** 2
        norm_y_sq = norm_y ** 2

        numerator = 0.5 * (norm_x_plus_y_sq - norm_x_sq - norm_y_sq)
        denominator = norm_x * norm_y + 1e-12  # avoid division by zero

        return (numerator / denominator).item()

    def calculate_policy(self):
        pi = np.zeros((self.num_users))
        r = np.zeros((self.num_users)) 
        pon = np.zeros((self.num_users))

        for iii in range(self.num_users):
            P10 = 1 - self.keepProbAvail[iii]
            P01 = 1 - self.keepProbNotAvail[iii]
            
            # Numerator
            term1 = (1 - P10)
            term2 = (P10 * P01) / (1 - P01)
            term3 = (P10 * P01) / ((1 - P01) ** 2) * np.log(P01)
            numerator = term1 - term2 - term3
            
            # Denominator
            denominator = 1 + P10 / P01
            
            r[iii] = numerator / denominator
            pon[iii] = P01/(P01 + P10)

        inverseSum = np.sum(r**(-1))
        pi = (r**(-1) / inverseSum)
        print(pi)
        print(pon)
        print(np.dot(pon, pi))

        pi = pi / np.dot(pon, pi) * self.bufferLimit


        return pi

    def innerProductTest(self):
        """" Inner Product Test from paper "" """
        if self.bufferSize == 0:
            return False
        choosenUsers = self.setAllUsers.difference(self.userListUL)
        
        global_grad_vector = torch.cat([(g/self.bufferSize).view(-1) for g in self.sum_terms])
        gradMag = torch.dot(global_grad_vector, global_grad_vector)
        print(self.bufferSize)
        varEst = 0
        for user in choosenUsers:
            user_grad_vector = torch.cat([(g/self.UserAgeMemory[user]).view(-1) for g in self.sparse_gradient[user]]).t()
            accInner = torch.dot(user_grad_vector, global_grad_vector)
            print(accInner/torch.norm(user_grad_vector)/torch.norm(global_grad_vector))
            varEst = varEst + torch.square(accInner-gradMag)
        varEst = varEst/max(1, self.bufferSize-1)
        
        conLHS = varEst/self.bufferSize
        conRHS = torch.square(self.theta_inner*gradMag)
        print("Inner Product Test:", conLHS, "<=", conRHS)
        check = conLHS <= conRHS
        return check 

    def orthogonalityTest(self):
        """" Inner Product Test from paper "" """
        if self.bufferSize == 0:
            return False
        choosenUsers = self.setAllUsers.difference(self.userListUL)

        global_grad_vector = torch.cat([(g/self.bufferSize).view(-1) for g in self.sum_terms])
        gradMag = torch.dot(global_grad_vector, global_grad_vector)

        orthTest = 0
        for user in choosenUsers:
            user_grad_vector = torch.cat([g.view(-1) for g in self.sparse_gradient[user]])
            accInner = torch.dot(user_grad_vector, global_grad_vector)
            grad = user_grad_vector - accInner/gradMag*global_grad_vector
            orthTest = orthTest + torch.dot(grad, grad)
        
        
        conLHS = orthTest/(max(1, self.bufferSize-1)*self.bufferSize)
        conRHS = (self.nu_orthogonal*self.nu_orthogonal)*gradMag
        print("Orthoganality Test:", conLHS, "<=", conRHS)

        check = conLHS <= conRHS 
        return check

    def updateAccuracy(self, accuracy):
        if accuracy > self.maxAccuracy:
            self.patience = 0
            self.maxAccuracy = accuracy
            self.bestWeight = [g.clone for g in self.w_global]
            

    def stepState(self):
        for iii in range(self.num_users):
            if (self.intermittentStateOneHot[iii]):
                self.intermittentStateOneHot[iii] = self.intermittentStateOneHot[iii] if self.keepProbAvail[iii] > random.random() else 1-self.intermittentStateOneHot[iii]
            else:
                self.intermittentStateOneHot[iii] = self.intermittentStateOneHot[iii] if self.keepProbNotAvail[iii] > random.random() else 1-self.intermittentStateOneHot[iii]
        self.intermittentUsers = np.where(self.intermittentStateOneHot)[0]

    # Calculate gradient difference between two sets of weights
    def calculate_gradient_difference(self, w_before, w_after):
        return [w_after[k] - w_before[k] for k in range(len(w_after))]
    
    # Sparsify the model weights
    def top_k_sparsificate_model_weights(self, weights, fraction):
        flat_weights = torch.cat([w.view(-1) for w in weights])
        threshold_value = torch.quantile(torch.abs(flat_weights), 1 - fraction)
        new_weights = []
        for w in weights:
            mask = torch.abs(w) >= threshold_value
            new_weights.append(w * mask.float())
        return new_weights
    

    def train_users(self, list_users):
        for user_id in list_users:

            # Reset model weights to the initial weights before each user's local training
            with torch.no_grad():
                for param, saved in zip(self.model.parameters(), self.w_user[user_id]):
                    param.copy_(saved) 
            torch.cuda.empty_cache()

            # Retrieve the user's training data (combined from all memory cells)
            trainloader = self.TrainSetUsers[user_id]
            
            if self.train_mode == "MNIST":
                for epoch in range(self.epochs):
                    for image, label in trainloader:
                        self.optimizer.zero_grad(set_to_none=True)     
                        image, label = image.to(self.device), label.to(self.device)  
                        output = self.model(image)
                        loss = self.criteron(output, label)
                        loss.backward()
                        self.optimizer.step()
                        torch.cuda.empty_cache()
            else: 
                for epoch in range(self.epochs): 
                    for image, label in trainloader:
                        self.optimizer.zero_grad(set_to_none=True)
                        image, label = image.to(self.device), label.to(self.device)  
                        output = self.model(image)
                        loss = self.criteron(output, label)
                        loss.backward()

                        self.optimizer.step()
        
            w_new = [param.data.clone() for param in self.model.parameters()]
            gradient_diff = self.calculate_gradient_difference(self.w_user[user_id], w_new)
            self.grad_per_user[user_id] = gradient_diff

            self.sparse_gradient[user_id] = self.top_k_sparsificate_model_weights(gradient_diff, self.fraction[0]) 

    def simulate_async_random(self, run, seed_index, timeframe):
        """Handles both Slotted ALOHA and standard user processing."""

        self.stepState()
        if (len(self.intermittentUsers) == 0):
            print("No users available passing")
            return self.w_global
        print(f"Available Users = {self.intermittentUsers.tolist()}")
        self.UserAgeUL = self.UserAgeUL + self.allOnes

        if (self.bufferSize == self.bufferLimit):
            user_grad_vector = torch.cat([g.view(-1) for g in self.sum_terms])
            global_grad_vector = torch.cat([g.view(-1) for g in self.lastSum_terms])

            # Compute cosine similarity
            self.lp_cos_val = self.lp_cosine_similarity(user_grad_vector, global_grad_vector, p = self.cos_similarity)
            print(f"Similarity of Gradients = {self.lp_cos_val}")
            self.w_global = [self.w_global[j] + self.sum_terms[j]/self.bufferSize for j in range(len(self.sum_terms))]
            self.lastSum_terms = self.sum_terms
            self.sum_terms = [torch.zeros_like(param).to(self.device) for param in self.w_global]
            self.bufferSize = 0
            self.UserAgeDL = self.UserAgeDL + self.allOnes

        selected_user_UL = random.sample(self.intermittentUsers.tolist(), 1)[0]
        print(f"Selected User in UL: {selected_user_UL}")
        self.train_users([selected_user_UL])
        self.sum_terms = [self.sum_terms[j] + self.sparse_gradient[selected_user_UL][j]/(self.UserAgeDL[selected_user_UL]+1) for j in range(len(self.sum_terms))]
        self.bufferSize = self.bufferSize + 1

        selected_user_DL = random.sample(self.intermittentUsers.tolist(), 1)[0]
        print(f"Selected user for downlink: {selected_user_DL}")
        self.w_user[selected_user_DL] = [w.clone() for w in self.w_global]

        for iii in range(len(self.w_global)):
            if torch.isnan(self.w_global[iii]).any():
                raise Exception("Nan on sparse gradient")
        
        self.UserAgeUL[selected_user_UL] = 0
        self.UserAgeDL[selected_user_DL] = 0
        print(f"User Age UL = {torch.t(self.UserAgeUL)}")
        print(f"User Age DL = {torch.t(self.UserAgeDL)}")

        return self.w_global
    
    def simulate_async_Age(self, run, seed_index, timeframe):
        """Handles both Slotted ALOHA and standard user processing."""

        self.UserAgeUL = self.UserAgeUL + self.allOnes
        self.stepState()
        if (len(self.intermittentUsers) == 0):
            print("No users available passing")
            return self.w_global
        print(f"Available Users = {self.intermittentUsers}")

        tempUserAgeDL = self.UserAgeDL[self.intermittentUsers]
        tempUserAgeUL = self.UserAgeUL[self.intermittentUsers]
        tempLastAge = self.lastAge[self.intermittentUsers]

        if (self.bufferSize == self.bufferLimit): 
            user_grad_vector = torch.cat([g.view(-1) for g in self.sum_terms])
            global_grad_vector = torch.cat([g.view(-1) for g in self.lastSum_terms])
            # Compute cosine similarity
            self.lp_cos_val = self.lp_cosine_similarity(user_grad_vector, global_grad_vector, p = self.cos_similarity)
            self.simQueue.append(self.lp_cos_val)
            print(f"Queue of Similarity of Gradients = {self.simQueue}")
            print(f"Average of Last 5 Similarity of Gradients = {np.mean(self.simQueue)}")
            print(f"Standard Deviation of Last 5 Similarity of Gradients = {np.std(self.simQueue)}")
            print(f"Similarity of Gradients = {self.lp_cos_val}")
            self.w_global = [self.w_global[j] + self.sum_terms[j]/self.bufferSize for j in range(len(self.sum_terms))] 
            for iii in range(len(self.w_global)):
                if torch.isnan(self.w_global[iii]).any():
                    raise Exception("Nan on aggregation")
            self.lastSum_terms = self.sum_terms
            self.sum_terms = [torch.zeros_like(param).to(self.device) for param in self.w_global]
            self.bufferSize = 0
            self.UserAgeDL = self.UserAgeDL + self.allOnes
            

        selected_user_UL = torch.argmax(tempUserAgeUL-tempUserAgeDL).item()
        selected_user_UL = self.intermittentUsers[selected_user_UL]
        print(f"Selected User in UL: {selected_user_UL}")
        self.train_users([selected_user_UL])
        self.sum_terms = [self.sum_terms[j] + self.sparse_gradient[selected_user_UL][j]/(self.UserAgeDL[selected_user_UL]+1) for j in range(len(self.sum_terms))] 
        for iii in range(len(self.sum_terms)):
            if torch.isnan(self.sum_terms[iii]).any():
                raise Exception("Nan on sum_terms")
        self.bufferSize = self.bufferSize + 1
        self.ageAcc = self.ageAcc + self.UserAgeDL[selected_user_UL]
        self.numUp = self.numUp + 1
        self.gradientMags.append(np.square(torch.norm(torch.stack([torch.norm(g/(self.UserAgeDL[selected_user_UL]+1)) for g in self.sparse_gradient[selected_user_UL]])).item()))

        selected_user_DL = torch.argmax(tempUserAgeDL).item()
        selected_user_DL = self.intermittentUsers[selected_user_DL]
        print(f"Selected User in DL: {selected_user_DL}")
        if self.UserAgeDL[selected_user_DL] > 0:
            self.w_user[selected_user_DL] = [w.clone() for w in self.w_global]
            self.UserAgeDL[selected_user_DL] = 0
        
        self.lastAge[selected_user_UL] = self.UserAgeDL[selected_user_UL]
        self.UserAgeUL[selected_user_UL] = 0


        print(f"User Age UL = {torch.t(self.UserAgeUL)}")
        print(f"User Age DL = {torch.t(self.UserAgeDL)}")
        print(f"Average Uplinked Model Age = {self.ageAcc/self.numUp}")

        return self.w_global

    def simulate_InnerProduct(self, run, seed_index, timeframe):
        """Handles both Slotted ALOHA and standard user processing."""
        self.UserAgeUL = self.UserAgeUL + self.allOnes

        tempUserAgeDL = self.UserAgeDL[self.intermittentUsers]
        tempUserAgeUL = self.UserAgeUL[self.intermittentUsers]
        tempLastAge = self.lastAge[self.intermittentUsers]
        
        self.stepState()
        if (len(self.intermittentUsers) == 0):
            print("No users available passing")
            return self.w_global
        print(f"Available Users = {self.intermittentUsers}")

        if (self.bufferSize >= 2) and (self.innerProductTest() and self.orthogonalityTest()) or (self.bufferSize == self.num_users):
            
            user_grad_vector = torch.cat([g.view(-1) for g in self.sum_terms])
            global_grad_vector = torch.cat([g.view(-1) for g in self.lastSum_terms])

            # Compute cosine similarity
            self.lp_cos_val = self.lp_cosine_similarity(user_grad_vector, global_grad_vector, p = self.cos_similarity)
            self.simQueue.append(self.lp_cos_val)
            
            print(f"Queue of Similarity of Gradients = {self.simQueue}")
            print(f"Average of Last 5 Similarity of Gradients = {np.mean(self.simQueue)}")
            print(f"Standard Deviation of Last 5 Similarity of Gradients = {np.std(self.simQueue)}")
            print(f"Similarity of Gradients = {self.lp_cos_val}")
            
            self.w_global = [self.w_global[j] + self.sum_terms[j]/self.bufferSize for j in range(len(self.sum_terms))]
            self.lastSum_terms = self.sum_terms
            self.sum_terms = [torch.zeros_like(param).to(self.device) for param in self.w_global]
            self.bufferSize = 0
            self.UserAgeDL = self.UserAgeDL + self.allOnes

            self.userListUL = set(range(self.num_users))
            self.userListDL = set(range(self.num_users))
            self.patience = self.patience + 1

        selected_user_UL = torch.argmax(tempUserAgeUL-tempUserAgeDL).item()
        selected_user_UL = self.intermittentUsers[selected_user_UL]
        self.userListUL.remove(selected_user_UL)
        print(f"Selected User in UL: {selected_user_UL}")
        self.train_users([selected_user_UL])
        self.UserAgeMemory[selected_user_UL] = self.UserAgeDL[selected_user_UL] + 1
        self.sum_terms = [self.sum_terms[j] + self.sparse_gradient[selected_user_UL][j]/self.UserAgeMemory[selected_user_UL] for j in range(len(self.sum_terms))]
        self.bufferSize = self.bufferSize + 1
    
        selected_user_DL = torch.argmax(tempUserAgeDL).item()
        selected_user_DL = self.intermittentUsers[selected_user_DL]
        self.userListDL.remove(selected_user_DL)
        print(f"Selected User in DL: {selected_user_DL}")
        self.w_user[selected_user_DL] = [w.clone() for w in self.w_global]

        self.lastAge[selected_user_UL] = self.UserAgeDL[selected_user_UL]
        self.UserAgeUL[selected_user_UL] = 0
        self.UserAgeDL[selected_user_DL] = 0


        print(f"User Age UL = {torch.t(self.UserAgeUL)}")
        print(f"User Age DL = {torch.t(self.UserAgeDL)}")

        return self.w_global
    
    def simulate_async_Asymp_EI(self, run, seed_index, timeframe):
        """Handles both Slotted ALOHA and standard user processing."""

        #New Available Users
        self.stepState()
        if (len(self.intermittentUsers) == 0):
            print("No users available passing")
            return self.w_global
        print(f"Available Users = {self.intermittentUsers}")

        #Choose available users according to their p_u
        tempPi = self.pi[self.intermittentUsers].flatten()            
        bernoulli_flips = np.random.rand(len(self.intermittentUsers)) < tempPi
        selected_users_UL = self.intermittentUsers[bernoulli_flips]
        self.num_send += len(selected_users_UL)
        if (len(selected_users_UL) == 0):
            print("No user transmits")
            return self.w_global
        print(f"Transmitting Users: {selected_users_UL.tolist()}")

        #Obtain gradient from users that transmit
        self.train_users(selected_users_UL.tolist())
        
        #Sum of trained gradients
        self.sum_terms = [torch.zeros_like(param).to(self.device) for param in self.w_global]
        for user in selected_users_UL:
            self.contribution[user] += 1/self.UserAgeDL[user].cpu().item()
            self.sum_terms = [self.sum_terms[j] + self.sparse_gradient[user][j]/(self.UserAgeDL[user]) for j in range(len(self.sum_terms))] 
        
        #Available users get the new global model
        for user in self.intermittentUsers:
            self.w_user[user] = [w.clone() for w in self.w_global]
            self.UserAgeDL[user] = 0

        self.w_global = [self.w_global[j] + self.sum_terms[j]/len(selected_users_UL) for j in range(len(self.sum_terms))] 
        self.UserAgeDL = self.UserAgeDL + self.allOnes

        return self.w_global
    
    def simulate_async_Asymp_Age(self, run, seed_index, timeframe):
        """Handles both Slotted ALOHA and standard user processing."""

        self.UserAgeUL = self.UserAgeUL + self.allOnes 
        
        #New Available Users
        self.stepState()
        if (len(self.intermittentUsers) == 0):
            print("No users available passing")
            return self.w_global
        print(f"Available Users = {self.intermittentUsers}")

        tempUserAgeUL = self.UserAgeUL[self.intermittentUsers]
        print(f"User Age UL: {tempUserAgeUL.squeeze()}")
        tempUserAgeDL = self.UserAgeDL[self.intermittentUsers] 
        print(f"User Age DL: {tempUserAgeDL.squeeze()}")

        # Calculate age difference and select top-k users
        age_diff = (tempUserAgeUL - tempUserAgeDL).squeeze()
        k = min(int(self.bufferLimit), len(self.intermittentUsers))
        
        sorted_indices = torch.argsort(age_diff, descending=True)
        topk_indices = sorted_indices[:k]
        
        selected_users_UL = self.intermittentUsers[topk_indices.cpu().numpy()]
        print(f"Selected User in UL: {selected_users_UL}")
        self.train_users(selected_users_UL.tolist())

        #Sum of trained gradients
        self.sum_terms = [torch.zeros_like(param).to(self.device) for param in self.w_global]
        for user in selected_users_UL:
            self.UserAgeUL[user] = 0
            self.contribution[user] += 1/self.UserAgeDL[user].cpu().item()
            self.sum_terms = [self.sum_terms[j] + self.sparse_gradient[user][j]/(self.UserAgeDL[user]) for j in range(len(self.sum_terms))] 
        
        #Available users get the new global model
        for user in self.intermittentUsers:
            self.w_user[user] = [w.clone() for w in self.w_global]
            self.UserAgeDL[user] = 0

        self.w_global = [self.w_global[j] + self.sum_terms[j]/len(selected_users_UL) for j in range(len(self.sum_terms))] 
        self.UserAgeDL = self.UserAgeDL + self.allOnes

        return self.w_global
    
    def simulate_test(self, run, seed_index, timeframe):
        self.train_users(list(range(self.num_users)))
        for user_id in range(self.num_users):
            for user_id2 in range(user_id, self.num_users):
                # Flatten gradients into 1D vectors
                user_grad_vector = torch.cat([g.view(-1) for g in self.grad_per_user[user_id]])
                global_grad_vector = torch.cat([g.view(-1) for g in self.grad_per_user[user_id2]])

                # Compute cosine similarity
                lp_cos_val = self.lp_cosine_similarity(user_grad_vector, global_grad_vector, p = self.cos_similarity)
                print(f"Similarity between {user_id} and {user_id2} = {lp_cos_val}")

    def simulate_transmissions(self):
        """Simulates slotted ALOHA transmissions."""
        decisions = np.random.rand(self.num_users) < self.tx_prob
        if np.sum(decisions) == 1:
            return [i for i, decision in enumerate(decisions) if decision]
        return []

    def run(self, runNo, seed_index, timeframe):
        """Dispatch based on the FL mode."""
        if self.mode == 'async_random':
            return self.async_random(runNo, seed_index, timeframe)
        elif self.mode == 'async_age':
            return self.async_age(runNo, seed_index, timeframe)
        elif self.mode == 'test':
            return self.test(runNo, seed_index, timeframe)
        elif self.mode == 'async_Inner':
            return self.async_InnerProduct(runNo, seed_index, timeframe)
        elif self.mode == 'async_asymp_EI':
            return self.simulate_async_Asymp_EI(runNo, seed_index, timeframe)
        elif self.mode == 'async_asymp_age':
            return self.simulate_async_Asymp_Age(runNo, seed_index, timeframe)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
 
    def async_random(self, run, seed_index, timeframe):
        print("Running Asynchronous Random")
        return self.simulate_async_random(run, seed_index, timeframe)
    
    def async_age(self, run, seed_index, timeframe):
        print("Running Asynchronous Age")
        return self.simulate_async_Age(run, seed_index, timeframe)
    
    def test(self, run, seed_index, timeframe):
        print("Running Test")
        return self.simulate_test(run, seed_index, timeframe)

    def async_InnerProduct(self, run, seed_index, timeframe):
        print("Running Asynchronous Inner Product Test")
        return self.simulate_InnerProduct(run, seed_index, timeframe)
    
    def async_Asymp_EI(self, run, seed_index, timeframe):
        print("Running Asynchronous Asymptotic Age")
        return self.simulate_async_Asymp_EI(run, seed_index, timeframe)
    
    def async_Asymp_Age(self, run, seed_index, timeframe):
        print("Running Asynchronous Asymptotic Age")
        return self.simulate_async_Asymp_Age(run, seed_index, timeframe)