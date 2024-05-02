import torch
import torch.nn as nn

class dehaze_net(nn.Module):

	def __init__(self):
		super(dehaze_net, self).__init__()

		self.relu = nn.ReLU(inplace=True)
		
		#nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding )
		self.e_conv1 = nn.Conv2d(3,3,1,1,0,bias=True) 
		self.e_conv2 = nn.Conv2d(3,3,3,1,1,bias=True)
		#in_channels = 6 because a concatenation outputs of e_conv1 and e_conv2 are used in e_conv3
		self.e_conv3 = nn.Conv2d(6,3,5,1,2,bias=True) 
		self.e_conv4 = nn.Conv2d(6,3,7,1,3,bias=True) 
		#in_channels = 12 because a concatenation outputs of e_conv2 and e_conv3 are used in e_conv3
		self.e_conv5 = nn.Conv2d(12,3,3,1,1,bias=True) 
		
	def forward(self, x):
		source = []
		source.append(x)

		x1 = self.relu(self.e_conv1(x))
		x2 = self.relu(self.e_conv2(x1))

		#concatenate x1 and x2
		concat1 = torch.cat((x1,x2), 1)
		x3 = self.relu(self.e_conv3(concat1))

		#concatenate x2 and x3
		concat2 = torch.cat((x2, x3), 1)
		x4 = self.relu(self.e_conv4(concat2))

		#concatenate x1, x2, x3, x4
		concat3 = torch.cat((x1,x2,x3,x4),1)
		x5 = self.relu(self.e_conv5(concat3))

		#J(X) = T(X)*I(X) - T(X) + B
		clean_image = self.relu((x5 * x) - x5 + 1) 
		
		#return J(x)
		return clean_image
