if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColorReset "${Esc}[m")
  set(ColourBold "${Esc}[1m")
  set(Underline "${Esc}[4m")
  set(Red "${Esc}[31m")
  set(Green "${Esc}[32m")
  set(Yellow "${Esc}[33m")
  set(Blue "${Esc}[34m")
  set(Magenta "${Esc}[35m")
  set(Cyan "${Esc}[36m")
  set(White "${Esc}[37m")
  set(BoldRed "${Esc}[1;31m")
  set(BoldGreen "${Esc}[1;32m")
  set(BoldYellow "${Esc}[1;33m")
  set(BoldBlue "${Esc}[1;34m")
  set(BoldMagenta "${Esc}[1;35m")
  set(BoldCyan "${Esc}[1;36m")
  set(BoldWhite "${Esc}[1;37m")
  set(DarkGray "${Esc}[1;90m")
  set(Dim "${Esc}[2m")
  set(StrikeBegin "${Esc}[9m")
  set(StrikeEnd "${Esc}[0m")
endif()

# message("This is normal") message("${Red}This is Red${ColorReset}")
# message("${Green}This is Green${ColorReset}") message("${Yellow}This is
# Yellow${ColorReset}") message("${Blue}This is Blue${ColorReset}")
# message("${Magenta}This is Magenta${ColorReset}") message("${Cyan}This is
# Cyan${ColorReset}") message("${White}This is White${ColorReset}")
# message("${BoldRed}This is BoldRed${ColorReset}") message("${BoldGreen}This is
# BoldGreen${ColorReset}") message("${BoldYellow}This is
# BoldYellow${ColorReset}") message("${BoldBlue}This is BoldBlue${ColorReset}")
# message("${BoldMagenta}This is BoldMagenta${ColorReset}")
# message("${BoldCyan}This is BoldCyan${ColorReset}") message("${BoldWhite}This
# is BoldWhite\n\n${ColorReset}")

# message()

