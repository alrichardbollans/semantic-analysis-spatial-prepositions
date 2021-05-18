# Annotator Agreement
Annotator agreements are calculated by process_data.py in Analysis and saved in the stats folder.

In each task agreement is calculated for each pair of Native English speaking users.

Cohens kappa is calculated for each pair of users and multiplied by the shared number of annotations. This is then summed for each user pair and divided by the total number of shared annotations. So the average Cohens Kappa given is weighted by shared annotations.

In both tasks observed agreement is the number of agreements divided by the number of shared annotations.

## Preposition Selection Task
A shared annotation in this task is where the users are shown the same figure and ground in the same scene.

As usual with Cohens Kappa expected agreement between users u1,u2 (for a given preposition) is the number of times in shared annotations that u1 says yes times the number of times that u2 says yes, plus number of times u1 says no times u2 says no divided by the number of shared annotations squared: `(y1*y2 + n1*n2))/(shared_annotations^2)`



## Comparative Task
A shared annotation in this task is where the users are shown the same ground and preposition in the same scene.

As we don't have category labels here we approximate expected agreement for a pair of users slightly differently.

Firstly we calculated the probability that u1 and u2 agree by both selecting none. This is the probability u1 selects none times probability that u2 selects none:
`u1_p_none = comp_none_selections1/shared_annotations`
`u2_p_none = comp_none_selections2/shared_annotations`
			

`expected_none_agreement = float(u1_p_none * u2_p_none)`

Then to work out the probability users agree on a particular object we calculate:

`average_probability_agree_on_object = (shared_annotations * (1-u1_p_none) * (1-u2_p_none))/number_of_compared_figures`

where number_of_compared_figures is the sum of all compared figures (potential objects to select) in all shared annotations.


Expected agreement is then simply:
`expected_agreement = expected_none_agreement + average_probability_agree_on_object`


See Agreements class for code.

