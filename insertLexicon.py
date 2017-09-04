from subprocess import check_output
from argparse import ArgumentParser
from datetime import datetime
import pandas as pd
import numpy as np
import re

_ANSI_COL_PATH = re.compile(b'\033\[[\d;]*m')

""" Change this to be the path to your local Graql directory """
_GRAQL_PATH = "/Users/nickpowell/Documents/grakn-newest/bin/graql.sh"

""" Global paths to relevant files """
_ONTOLOGY_PATH = "wordnetOntology.gql"
_RULESET_PATH = "wordnetRules.gql"
_ENTITIES_OUTPUT_PATH = "entityinsert.gql"
_RELATIONS_OUTPUT_PATH = "relationsinsert.gql"
_TESTSET_OUTPUT_PATH = "testresults.gql"
_INFERENCE_CHECK_PATH = "inferencecheck.gql"

ontology_relation_map = {
		'_type_of'					:	'type-of',
		'_synset_domain_topic'		:	'synset-domain-topic',
		'_has_instance'				:	'has-instance',
		'_part_of'					:	'part-of',
		'_has_part'					:	'has-part',
		'_member_holonym'			:	'holonym',
		'_member_meronym'			:	'meronym',
		'_similar_to'				:	'similar-to',
		'_subordinate_instance_of'	:	'subordinate-instance-of',
		'_domain_topic'				:	'domain-topic',
		'_domain_region'			:	'domain-region'
	}

ontology_role_map = {
		'_type_of' 					:	['type-left', 'type-right'],
		'_synset_domain_topic' 		:	['syndomaintopic-left', 'syndomaintopic-right'],
		'_has_instance'				:	['instance-left', 'instance-right'],
		'_part_of'					:	['partof-left', 'partof-right'],
		'_has_part'					:	['haspart-left', 'haspart-right'],
		'_member_holonym'			:	['holonym-left', 'holonym-right'],
		'_member_meronym'			:	['meronym-left', 'meronym-right'],
		'_similar_to'				:	['similar-left', 'similar-right'],
		'_subordinate_instance_of'	:	['subinstance-left', 'subinstance-right'],
		'_domain_topic'				:	['domaintopic-left', 'domaintopic-right'],
		'_domain_region'			:	['domainregion-left', 'domainregion-right']
}

reciprocal_relations_map = {
		'_type_of'					:	'_has_instance',
		'_subordinate_instance_of'	:	'_has_instance',
		'_synset_domain_topic'		:	'_domain_region',
		'_domain_region'			:	'_synset_domain_topic',
		'_part_of'					:	'_has_part',
		'_has_part'					:	'_part_of',
		'_member_holonym'			:	'_member_meronym',
		'_member_meronym'			:	'_member_holonym'
}

###########################################################################################
""" Gets the reciprocal relation of the parameter relation, if it has a reciprocal relation """
def getReciprocal(relation):
	if relation in reciprocal_relations_map:
		return reciprocal_relations_map[relation]
	else:
		return None


###########################################################################################
""" Creates ontology from the files at the global paths """
def insertOntology(keyspace):
	check_output(_GRAQL_PATH + " -f " + _ONTOLOGY_PATH + " -k " + keyspace, shell=True)
	check_output(_GRAQL_PATH + " -f " + _RULESET_PATH + " -k " + keyspace, shell=True)


###########################################################################################
""" Inserts entities into existing ontology """
def insertEntities(keyspace, filename):
	file_object = open(filename, 'r')
	data = file_object.read().splitlines()
	query = ""
	for word in data:
		entity_name = "\"" + str(word) + "\""
		query += '''insert $x isa word has name ''' + entity_name + '''; '''

	""" """
	f = open(_ENTITIES_OUTPUT_PATH, 'w')
	f.write(query)
	f.close()
	result = check_output(_GRAQL_PATH + " -b " + _ENTITIES_OUTPUT_PATH + " -k " + keyspace, shell=True)

###########################################################################################
""" Takes a 'batch' of data and forms a Graql query out of it """
def addBatchToQuery(batch, is_test):
	query = ""
	for test_triplet in batch:

		""" Each triplet is in the form (e1, R, e2) as described in the paper """
		triplet = test_triplet.split('\t')
		e1 = "\"" + triplet[0] + "\""
		e2 = "\"" + triplet[2] + "\""
		relation = ontology_relation_map[triplet[1]]
		roles = ontology_role_map[triplet[1]]
		if not is_test:
			query += '''match $x isa word has name ''' + e1 + '''; $y isa word has name ''' + e2 + '''; insert (''' + roles[0] + ''': $x, ''' + roles[1] + ''': $y) isa ''' + relation + '''; '''
		else:
			query += '''match $x isa word has name ''' + e1 + '''; $y isa word has name ''' + e2 + '''; (''' + roles[0] + ''': $x, ''' + roles[1] + ''': $y) isa ''' + relation + '''; ask; '''
	return query


###########################################################################################
""" Inserts the training relations from filename into the keyspace """
def insertRelations(keyspace, filename):
	file_object = open(filename, 'r')
	data = file_object.read().splitlines()
	batch_size = 10000
	running_batch_count = 0
	batch = []
	print "---> Beginning relation insert "

	## inserting every relation at once may cause an OutOfMemory exception, so we divide the set into batches
	while batch_size * running_batch_count <= len(data):
		if batch_size*(running_batch_count+1) <= len(data):
			batch = data[running_batch_count*batch_size:(running_batch_count+1)*batch_size]
		else:
			batch = data[running_batch_count*batch_size:]

		## isTest variable set to False indicating that we are adding relations, not checking them
		query = addBatchToQuery(batch, False)
		f = open(_RELATIONS_OUTPUT_PATH, 'w')
		f.write(query)
		f.close()
		result = check_output(_GRAQL_PATH + " -b " + _RELATIONS_OUTPUT_PATH + " -k " + keyspace, shell=True)
		running_batch_count += 1
		print "     Running batch count is " + str(running_batch_count)

	print "     Relation insert finished"


###########################################################################################
""" Checks the Graql graph for inference relations at the specified indices of 'test.txt' """
def checkGraqlGraph(keyspace, indices):

	start_check = datetime.now()
	file_object = open('test.txt', 'r')
	data = file_object.read().splitlines()

	""" Takes only the subset of relevant indices that we want to check in Grakn """
	if indices:
		data = [data[i] for i in indices]

	batch_size = 100
	batch = []
	running_batch_count = 0
	total_batch_count = (len(data) / batch_size) + 1

	""" Initialize predictions as an empty list """
	predictions = []

	start = datetime.now()
	print "---> Starting batching "

	""" Takes the data and breaks it into chunk queries """
	while batch_size * running_batch_count <= len(data):
		if batch_size*(running_batch_count+1) <= len(data):
			batch = data[running_batch_count*batch_size:(running_batch_count+1)*batch_size]
		else:
			batch = data[running_batch_count*batch_size:]

		start_batch = datetime.now()
		query = addBatchToQuery(batch, True)

		batch_query = "\'" + query + "\'"
		start_batch = datetime.now()
		result = check_output(_GRAQL_PATH + " -e " + batch_query + " -n -k " + keyspace, shell=True)
		end_batch = datetime.now()
		diff = end_batch-start_batch

		""" Do string parsing on the result string returned by Grakn, deleting the last, empty item from the list """
		result_string = re.sub(_ANSI_COL_PATH, '', result)
		result_string = result_string.split('\n')[:-1]

		predictions.extend(result_string)

		running_batch_count += 1
		print "------> Finished " + str(running_batch_count) + " out of " + str(total_batch_count) + " batches (most recent batch took " + str((end_batch-start_batch).total_seconds()) + " seconds)      \r",

	arr_results = np.array([x=='True' for x in predictions])

	np_pred = np.asarray(predictions, dtype=np.object)

	np_pred[np_pred == 'True'] = 1
	np_pred[np_pred == 'False'] = -1

	end_check = datetime.now()
	print "------> " + str((end_check-start_check).total_seconds()) + " total seconds spent searching the graph                                                                     "

	return np_pred


###########################################################################################
""" Main function if insertLexicon needs to be run alone"""
if __name__=="__main__":
	parser = ArgumentParser(
		description="insertLexicon -k KEYSPACE")
	parser.add_argument('-k', '--keyspace', help="The Graql keyspace to use", required=True)
	sysargs = parser.parse_args()

	print "Creating ontology and ruleset..."
	insertOntology(sysargs.keyspace)

	print "Inserting entities into graph..."
	insertEntities(sysargs.keyspace, 'entities.txt')

	print "Inserting relations into graph..."
	insertRelations(sysargs.keyspace, 'train.txt')

	print "Checking graph for inferred relations..."
	checkGraqlGraph(sysargs.keyspace, [])
	




