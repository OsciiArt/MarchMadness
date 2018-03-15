from bracketeer import build_bracket
b = build_bracket(
        outputPath='../output/Moutput.png',
        teamsPath='../Minput2/Teams.csv',
        seedsPath='../Minput3/NCAATourneySeeds.csv',
        submissionPath='../output/Mstats_2018_AB.csv',
        slotsPath='../Minput3/NCAATourneySlots.csv',
        year=2018
)