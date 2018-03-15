from bracketeer import build_bracket
b = build_bracket(
        outputPath='../output/Woutput.png',
        teamsPath='../Winput2/WTeams.csv',
        seedsPath='../Winput2/WNCAATourneySeeds.csv',
        submissionPath='../output/Wstats_2018_AB.csv',
        slotsPath='../Winput2/WNCAATourneySlots.csv',
        year=2018
)