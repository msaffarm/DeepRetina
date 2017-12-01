# plotting results
import numpy as np
import os
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

class plotter():

	def __init__(self, loggerList, numOfEpochs, outDir):
		self.loggerList = loggerList
		self.numOfEpochs = numOfEpochs
		self.outDir = outDir

	def updateLogger(self,data, label):
		# find logger and add data to it
		for logger in self.loggerList:
			if logger.getLabel()==label:
				logger.addData(data)



	def saveLoggers(self):
		if not os.path.lexists(self.outDir):
			os.makedirs(self.outDir)
		#  plot loggers
		with open(self.outDir + "/results.txt","w") as output:
			for logger in self.loggerList:
				output.write(logger.getLabel())
				output.write(" ")
				for val in logger.getData():
					output.write(str(val) + " ")
				output.write("\n")


	def plotLoggers(self):
		# make plot dir if not exists
		if not os.path.lexists(self.outDir):
			os.makedirs(self.outDir)
		#  plot loggers
		for logger in self.loggerList:
			xmin, xmax = 0, self.numOfEpochs-1
			
			xlabel = "update batches"
			ylabel = logger.getLabel()
			f, ax = matplotlib.pyplot.subplots()
			ax.set_autoscale_on(False)
			ax.set_xlim(left=xmin, right=xmax)
			
			if logger.getLabel().split()[0]=="validation" or \
			logger.getLabel().split()[0]=="dice":
				data = logger.getData()[1::]
				ymin, ymax = 0, max(data)
				ax.set_ylim(bottom=ymin, top=ymax)
				ax.plot(range(self.numOfEpochs),data,color=logger.getColor())
			else:
				data = logger.getData()
				ymin, ymax = 0, max(data)
				ax.set_ylim(bottom=ymin, top=ymax)
				ax.plot(range(self.numOfEpochs), data,color=logger.getColor())

			fig_name = "".join(list(x + "_" for x in logger.getLabel().split())) 
			matplotlib.pyplot.xlabel(xlabel)
			matplotlib.pyplot.ylabel(ylabel)
			matplotlib.pyplot.title(fig_name)
			matplotlib.pyplot.legend()
			matplotlib.pyplot.savefig(self.outDir + "/" + fig_name)
		
		matplotlib.pyplot.show()
		matplotlib.pyplot.close()

# logging different loss and erros in each model

class logger():

	def __init__(self,data = [], label = "NoName!",color = "Red"):
		self.data = []
		self.label = label
		self.color = color


	def addData(self,d):
		self.data.append(d)


	def getData(self):
		return self.data

	def getLabel(self):
		return self.label

	def getColor(self):
		return self.color







def main():
	pass


if __name__ == '__main__':
	main()