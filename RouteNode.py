from anytree import *
import datetime
import numpy as np

class RouteNode(NodeMixin):
	def __init__(self, name, trip_id=None, pickup=None, loc=[0,0], route_opt=None, trip_time=0, parent=None):
		super(NodeMixin, self).__init__()
		self.name = name
		self.trip_id = trip_id
		self.pickup = pickup # boolean: True=origin/pickup, False=destination/dropoff
		self.loc = loc
		self.parent = parent # previous node in route
		
		if self.parent is not None:
			self.total_time = self.parent.total_time + trip_time
		else:
			self.total_time = trip_time
		
		self.route_options = route_opt # dictionary of trips storing information about who can be picked up/dropped off
		
		if self.pickup is True: # picking up, add 1 to total number of people in car
			self.num_in_car = self.parent.num_in_car + 1
		elif self.pickup is False: # dropping off, subtract 1 from total number of people in car
			self.num_in_car = self.parent.num_in_car - 1
		else:
			self.num_in_car = 0
	
	def route_to_node(self, datetime_zero):
		'''Traverse route backwards starting from node; for printing.'''
		time_of_day = datetime_zero + datetime.timedelta(seconds=self.total_time*60)
		if self.parent is not None:
			return self.parent.route_to_node(datetime_zero) + time_of_day.strftime('%H:%M:%S') + ' > %s @ [%i, %i]\n' % (self.name, self.loc[0], self.loc[1])
		else:
			return ''