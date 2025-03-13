import torch
class matrix_utils():

    def block_diag_matrix_builder(A,B):
        # Builds a block matrix of the form [[A, 0], [0, B]]
        # A and B must be compatible tensors
        
        print(f"block_diag_matrix_builder ENTRY: A shape: {A.shape}, B shape: {B.shape}")
        
        # Special case: If B has a second dimension of 0, return A directly
        if len(B.shape) > 1 and B.shape[-1] == 0:
            print(f"block_diag_matrix_builder: B has second dimension of 0, returning A with shape {A.shape}")
            # We should simply return A without modifying it
            return A
        
        # Special case: If A has a second dimension of 0, return B directly
        if len(A.shape) > 1 and A.shape[-1] == 0:
            print(f"block_diag_matrix_builder: A has second dimension of 0, returning B with shape {B.shape}")
            # We should simply return B without modifying it
            return B
            
        # Special case: If either A or B has 0 in any other dimension, handle carefully
        if 0 in A.shape[:-1] or 0 in B.shape[:-1]:
            print(f"block_diag_matrix_builder: A or B has 0 in shape dimensions: A={A.shape}, B={B.shape}")
            # Handle appropriately based on which tensor has zero dimensions
            if 0 in A.shape[:-1]:
                return B
            else:
                return A
        
        # Get the shapes of A and B
        A_shape = A.shape
        B_shape = B.shape
        
        # Create zero tensors for the off-diagonal blocks
        zeros_top_right = torch.zeros(A_shape[:-1] + (B_shape[-1],), device=A.device)
        zeros_bottom_left = torch.zeros(B_shape[:-1] + (A_shape[-1],), device=B.device)
        
        # Concatenate the top row and bottom row
        top_row = torch.cat((A, zeros_top_right), dim=-1)
        bottom_row = torch.cat((zeros_bottom_left, B), dim=-1)
        
        # Concatenate the rows to form the block matrix
        block_matrix = torch.cat((top_row, bottom_row), dim=-2)
        print(f"block_diag_matrix_builder EXIT: block_matrix shape: {block_matrix.shape}")
        
        return block_matrix

    def block_matrix_inverse(A,B,C,D,block_form=True):
            # inverts a block matrix of the form [A B; C D] and returns the blocks [Ainv Binv; Cinv Dinv]
        invA = A.inverse()
        invD = D.inverse()
        Ainv = (A - B@invD@C).inverse()
        Dinv = (D - C@invA@B).inverse()
        
        if(block_form == 'left'):     # left decomposed returns abcd.inverse = [A 0; 0 D] @ [eye B; C eye]
            return Ainv, -B@invD, -C@invA, Dinv
        elif(block_form == 'right'):  # right decomposed returns abcd.inverse =  [eye B; C eye] @ [A 0; 0 D]
            return Ainv, -invA@B, -invD@C, Dinv            
        elif(block_form == 'True'):
            return Ainv, -Ainv@B@Dinv, -invD@C@invA, Dinv
        else:
            return torch.cat((torch.cat((Ainv, -invA@B@Dinv),-1),torch.cat((-invD@C@Ainv, Dinv),-1)),-2)

    def block_matrix_builder(A,B,C,D):
        # builds a block matrix [[A,B],[C,D]] out of compatible tensors
        return torch.cat((torch.cat((A, B),-1),torch.cat((C, D),-1)),-2)

    def block_precision_marginalizer(A,B,C,D):
        # When computing the precision of marginals, A - B@invD@C, does not need to be inverted
        # This is because (A - B@invD@C).inverse is the marginal covariance, the inverse of which is precsion
        # As a result in many applications we can save on computation by returning the inverse of Joint Precision 
        # in the form [A_prec 0; 0 D_prec] @ [eye B; C eye].  This is particularly useful when computing 
        # marginal invSigma and invSigmamu since    invSigma_A = A_prec
        #                                         invSigmamu_A = invSigmamu_J_A - B@invD@invSigmamu_J_B
        #                                           invSigma_D = D_prec
        #                                         invSigmamu_D = invSigmamu_J_D - C@invA@invSigmamu_J_A
         
        invA = A.inverse()
        invD = D.inverse()
        A_prec = (A - B@invD@C)
        D_prec = (D - C@invA@B)

        return A_prec, -B@invD, -C@invA, D_prec


    def block_matrix_logdet(A,B,C,D,singular=False):
        if(singular == 'A'):
            return D.logdet() + (A - B@D.inverse()@C).logdet()
        elif(singular == 'D'):            
            return A.logdet() + (D - C@A.inverse()@B).logdet()
        else:
            return D.logdet() + (A - B@D.inverse()@C).logdet()

