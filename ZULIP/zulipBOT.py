import zulip
import random
import time

clientStud = zulip.Client(email="eliza-2014-bot@students.hackerschool.com",
	api_key="oF9ZOQpkMSaPZl3jWAuZBfhcK7of0eEH", verbose=True)

clientProf = zulip.Client(email="prof-2014-bot@students.hackerschool.com",
	api_key="X9ugIz7wZS0h5edPlSpDC0kXNyvbvOQj")

def run():
	def do_register():
		registerStud = clientStud.register(event_types=['message'])
		registerProf = clientProf.register(event_types=['message'])
		q_id = [registerStud['queue_id'], registerProf['queue_id']]
		last_event_id = [registerStud['last_event_id'], registerProf['last_event_id']]
		return [q_id, last_event_id]

	q_id = None
	send_first = False
	while True:
		print 'starting loop'
		if q_id is None:
			q_id, last_event_id = do_register()
			#print q_id, last_event_id

		if send_first == False:
			# eliza-bot sends a message to start conversation
			clientStud.send_message({
				"type": "stream",
				"to": "test-bot2",
				"subject": 'algorithms - help!',
				"content": "My algorithms class is twisting my brain. I need help."
				})
			send_first = True

		res = clientProf.get_events(queue_id = q_id[1], last_event_id=last_event_id[1])

		if 'error' in res.get('result'):
			if res["result"] == "http-error":
				print "HTTP error fetching events -- probably a server restart"
			elif res["result"] == "connection-error":
				print "Connection error fetching events -- probably server is temporarily down?"
			else:
				print "Server returned error:\n%s" % res["msg"]
		else:
			for event in res['events']:
				#print event['message']['sender_short_name']
				last_event_id[1] = max(last_event_id[1], int(event['id']))
				processEventProf(event)

		res = clientStud.get_events(queue_id = q_id[0], last_event_id=last_event_id[0])
		if 'error' in res.get('result'):
			if res["result"] == "http-error":
				print "HTTP error fetching events -- probably a server restart"
			elif res["result"] == "connection-error":
				print "Connection error fetching events -- probably server is temporarily down?"
			else:
				print "Server returned error:\n%s" % res["msg"]
		else:
			for event in res['events']:
				last_event_id[0] = max(last_event_id[0], int(event['id']))
				# parse event and send new event to stream
				if event['type'] == 'message':
					print 'student just got: {}'.format(event['message']['content'])
				else:
					print 'got event of type {}'.format(event['type'])
				processEventStud(event)

# student knowledgebase
baseStud = {}
baseStud['visual'] = ''

baseStud['algor'] = ['There so many algorithms for sorting. I am overwhelmed.', 
'How can I get all this algorithm stuff straight in my head?',
'I think I get the algorithm stuff until I try and do the graph homework.']

baseStud['sort'] = ['Can you just tell me which sorting routine is best?', 
'How do I know which sorting routine to use?',
'I find it difficult to understand the benefits of each type of radix sort.',
'What are the fundamental things I need to know to sort?',
'I think I like graphing better than sorting.',
'Graphing seems more intuitive.']

baseStud['graph'] = ['I find it hard to remember which graph routine uses which graph type. Any tips on how to get this straight?',
'Max Flow graph processing was cool. One day I want to reduce a random problem to a max flow graph.',
'I need more examples of how strings can be represented by graphs.',
'The slide on LSD sorting was good. I still feel insecure though.',
'I need to understand why union-find is needed for MST. The marked array is not enough.',
'In lecutre you said "If you were paying attention, you should see that Prim and Dijkstra graph processing are similiar. I was paying attention!']

baseStud['data'] = ''
baseStud['review'] = ''
baseStud['funct'] = ''
baseStud['code'] = ''
def processEventStud(event):
	'''student's knowledge base'''
	event_type = event["type"]
	event_id = event['id']
	if 'message' in event:
		# parse
		msg = event['message']
		msg_type = msg['type']  			# private or stream
		name = msg['display_recipient']		# stream name
		topic = msg['subject'] 				# stream topic
		sender = msg['sender_full_name']
		content = msg['content']
		print 'student receives from', sender
		print '{}\n'.format(content)
		# student responds to stream
		if msg_type == 'stream' and name == 'test-bot2':
			print 'successfully identified msg_type, name'
			print 'sender is', sender, 'expecting prof-bot'
			if sender == 'prof-bot':
				print 'successfully identified sender'
				sent = False
				for key in baseStud:
					if key in content:
						sent = True
						clientStud.send_message({
							"type": "stream",
							"to": "test-bot2",
							"subject": 'algorithms - help!',
							"content": random.choice(baseStud[key])
							})
						time.sleep(1)
						# print 'stud sent resp'
		else:
			clientStud.send_message({
							"type": "stream",
							"to": "test-bot2",
							"subject": 'algorithms - help!',
							"content": "Sometimes I feel so down when I can't answer those sorting problems."
							})
			time.sleep(1)

				# if not sent:
				# 	raise Exception('freak out!')


# knowledgebase
baseProf = {}
baseProf['algor'] = ['That is a little too general. Do you need help with sorting or graphs?.', 
'If you get good, you can do research with me on graph processing.',
'Do you need help with graphs?', 'You might need to buy my book. Amazon Prime. 2-day delivery.',
'If you want a job, work on the graph interview questions. Pretty tricky.',
'I actually like radix sorts. Reminds me of the 70s.',
'The Fourth Edition of my book got graph processing correct. Took us 4 tries.']

baseProf['sort'] = ['selection, insertion, merge, quick, 3-way quick, LSD radix, MSD radix, MSD+quick radix',
'Do some experiments: generate random decimal keys, random NJ license plates, random fixed length words.',
'The first step is to understand how your programming language implements strings.',
'Understanding key-indexed counting is the first step for understanding string sorting. See p. 705 of my book.']

baseProf['graph'] = ['Did you say graph? I love teaching about graphs!', 
'Why do we have so many graph classes? Just so each one can be implemented efficiently. May be better to have one parent graph class.',
'It is important to understand the difference between Dijkstra and A Star graph searches.',
'Find the shortest path in a Euclidean graph. You will understand A Star better. The answer is not in my book. (Ha Ha)',
'Make sure you understand index priority queues. Very important in processing graphs with Dijkstra.',
'BFS, DFS, Dijkstra, Prim, Kruskal are in the same family of search algorithms. Make sure you understand why. I left this out of my book.',
'Make sure you understand why Dijkstra and Prim"s algorithm are basically the same.',
'Remember an undirected graph is a directed graph, where every edge is bidirectional.',
'You must understand LSD string sort for the exam.',
'Sorting is one of my favorite topics, even more than graph processing.']

def processEventProf(event):
	'''prof's knowledge base'''
	event_type = event["type"]
	event_id = event['id']
	if 'message' in event:
		# parse
		msg = event['message']
		msg_type = msg['type']  			# private or stream
		name = msg['display_recipient']		# stream name
		topic = msg['subject'] 				# stream topic
		sender = msg['sender_full_name']
		content = msg['content']
		print 'prof receives from', sender
		print '{}\n'.format(content)

		if msg_type == 'stream' and name == 'test-bot2':
			#print sender
			if sender == 'eliza-bot':
				for key in baseProf:
					#print key
					if key in content:
						clientProf.send_message({
							"type": "stream",
							"to": "test-bot2",
							"subject": 'algorithms - help!',
							"content": random.choice(baseProf[key])
							})
						time.sleep(1)
						# print 'prof sends message'
	else:
		# no more messages on queue
		clientProf.send_message({
							"type": "stream",
							"to": "test-bot2",
							"subject": 'algorithms - help!',
							"content": "You really should buy my hardcover book. These are the coolest algorithms ever discovered, and that undergrads can understand."
							})


run()



